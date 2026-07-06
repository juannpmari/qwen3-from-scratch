import os
import torch
import torch.nn as nn
import torch.distributed as dist


def setup_distributed():
    """Called once at process startup, before building the model.
    torchrun sets RANK / LOCAL_RANK / WORLD_SIZE env vars for us."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Backend choice: nccl needs real GPUs; gloo is its CPU cousin. On your
    # single-GPU box we emulate multiple ranks as CPU processes with gloo.
    # Force explicitly with TP_BACKEND=gloo (or nccl); otherwise auto-detect.
    backend = os.environ.get("TP_BACKEND")
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend)
    if backend == "nccl":
        torch.cuda.set_device(local_rank)  # local_rank==0 for everyone here
    return rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


def tp_rank():
    return dist.get_rank()


def tp_world():
    return dist.get_world_size()


def tp_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def tp_device():
    """Where this rank's tensors live. Tied to the backend: gloo → CPU (our
    multi-process CPU emulation), nccl → this rank's GPU."""
    if dist.is_initialized() and dist.get_backend() == "gloo":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{tp_local_rank()}")
    return torch.device("cpu")


def sync_replicated_params(model, src=0):
    """Make every REPLICATED weight identical across ranks by broadcasting from
    rank `src`. Each process built the model with its own RNG, so replicated
    params (embeddings, norms, output layer) started out different — this fixes
    that. SHARDED params are skipped: each rank must keep its own distinct shard.

    Call this ONCE, right after constructing the model."""
    for p in model.parameters():
        if getattr(p, "tp_sharded", False):
            continue  # column/row-parallel shard — intentionally differs per rank
        dist.broadcast(p.data, src=src)
    # buffers (e.g. the causal mask) are deterministic, but broadcasting is cheap
    # insurance that every rank is byte-identical on the replicated state.
    for b in model.buffers():
        dist.broadcast(b.data, src=src)


# ---------------------------------------------------------------------------
# Two ways to get replicated params consistent across ranks. Pick with `mode`:
#
#   "broadcast": every rank random-inits everything, then rank 0's replicated
#                params are broadcast to all. Simple and explicit, but ranks
#                1..N-1 waste their replicated-param init (it gets overwritten).
#
#   "dual_rng":  build the whole model under a SHARED seed so replicated params
#                come out identical on every rank with no communication; then
#                re-init only the SHARDED params with a per-rank seed so each
#                rank gets a distinct shard. No wasted init, no broadcast.
#
# Usage (bracketing model construction):
#     seed_model_init(mode, shared_seed)      # before building
#     model = Transformer(...); model.to(...)
#     finalize_model_init(model, mode, shared_seed)   # after building
# ---------------------------------------------------------------------------


def _reinit_sharded_params(model, seed):
    """Re-draw ONLY the sharded weights using a per-rank seed → each rank gets a
    distinct shard. Replicated params are left untouched. We temporarily hijack
    the global RNG and restore it afterwards so nothing downstream is perturbed."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    torch.manual_seed(seed)  # this rank's private init stream (CPU + CUDA)
    for module in model.modules():
        if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
            module.reset_parameters()

    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def seed_model_init(mode, shared_seed=0):
    """Call BEFORE constructing the model.

    dual_rng: seed the global RNG so every rank builds IDENTICAL params (the
              sharded ones are fixed up later in finalize_model_init).
    broadcast: no-op — each rank keeps its own entropy-based random init."""
    if mode == "dual_rng":
        torch.manual_seed(shared_seed)
    elif mode == "broadcast":
        pass
    else:
        raise ValueError(f"unknown TP init mode: {mode!r}")


def finalize_model_init(model, mode, shared_seed=0):
    """Call AFTER constructing the model and moving it to its device.

    dual_rng: give each rank a distinct shard (replicated params already agree).
    broadcast: broadcast rank 0's replicated params to everyone."""
    if mode == "dual_rng":
        _reinit_sharded_params(model, seed=shared_seed + 1 + tp_rank())
    elif mode == "broadcast":
        sync_replicated_params(model)
    else:
        raise ValueError(f"unknown TP init mode: {mode!r}")


# ---------------------------------------------------------------------------
# The f and g operators (Megatron's "conjugate" pair).
#
# These are the ONLY place communication happens. They are torch.autograd
# Functions, so we write the forward comm once and autograd inserts the
# backward comm for us automatically.
# ---------------------------------------------------------------------------


class _F(torch.autograd.Function):
    """f = {forward: identity, backward: all_reduce}.

    Sits at the INPUT of a column-parallel region. The input x is replicated
    on every rank, so forward does nothing. In backward, each rank computes a
    partial gradient w.r.t. that replicated input, and they must be summed."""

    @staticmethod
    def forward(ctx, x):
        return x  # identity — x is already the same on every rank

    @staticmethod
    def backward(ctx, grad):
        # sum the input-gradient across all ranks; all-reduce is in-place
        grad = grad.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)  # op defaults to SUM
        return grad


class _G(torch.autograd.Function):
    """g = {forward: all_reduce, backward: identity}.

    Sits at the OUTPUT of a row-parallel region. Each rank produced a PARTIAL
    output (its slice of the sum), so forward must all-reduce them into the
    full result. In backward, the incoming grad is already the same on every
    rank, so each rank just passes it through."""

    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)  # sum the partial outputs → full output
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad  # identity


# convenience wrappers so call sites read like functions
def f(x):
    return _F.apply(x)


def g(x):
    return _G.apply(x)


# ---------------------------------------------------------------------------
# The two parallel Linear layers. Both mirror your src/common/linear.py:
# weight has shape (out_features, in_features) and forward is x @ weight.t().
# ---------------------------------------------------------------------------


class ColumnParallelLinear(nn.Module):
    """Splits the weight along its OUTPUT dimension.

    Each rank holds out_features/world rows of the weight and therefore
    produces a SHARD of the output columns. No comm in forward; the f operator
    handles the backward all-reduce on the (replicated) input gradient.

    Input:  x           replicated,  (..., in_features)
    Output: y_shard     sharded,     (..., out_features // world)
    """

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        world = tp_world()
        assert out_features % world == 0, "out_features must divide world size"
        self.in_features = in_features
        self.out_features_per_rank = out_features // world

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features, device=device, dtype=dtype)
        )
        self.weight.tp_sharded = True  # skip in sync_replicated_params
        # This weight is split along dim 0 (output rows). Consolidated checkpoint
        # save/load uses shard_dim to know which axis to gather/slice on.
        self.shard_dim = 0
        # init variance uses the FULL out_features, matching the single-GPU Linear
        self._init_std = (2 / (in_features + out_features)) ** 0.5
        self.reset_parameters()

    def reset_parameters(self):
        # factored out so dual-RNG init can re-draw this shard with a per-rank seed
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=self._init_std,
            a=-self._init_std * 3.0,
            b=self._init_std * 3.0,
        )

    def forward(self, x):
        x = f(x)  # identity now; all-reduce of dx in backward
        return x.matmul(self.weight.t())  # local shard of the output


class RowParallelLinear(nn.Module):
    """Splits the weight along its INPUT dimension.

    Each rank holds in_features/world columns of the weight and consumes the
    matching SHARD of the input, producing a PARTIAL output. The g operator
    all-reduces those partials into the full output.

    Input:  x_shard     sharded,     (..., in_features // world)
    Output: y           replicated,  (..., out_features)
    """

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        world = tp_world()
        assert in_features % world == 0, "in_features must divide world size"
        self.in_features_per_rank = in_features // world
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank, device=device, dtype=dtype)
        )
        self.weight.tp_sharded = True  # skip in sync_replicated_params
        # This weight is split along dim 1 (input columns). Consolidated
        # checkpoint save/load uses shard_dim to gather/slice on the right axis.
        self.shard_dim = 1
        # init variance uses the FULL in_features, matching the single-GPU Linear
        self._init_std = (2 / (in_features + out_features)) ** 0.5
        self.reset_parameters()

    def reset_parameters(self):
        # factored out so dual-RNG init can re-draw this shard with a per-rank seed
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=self._init_std,
            a=-self._init_std * 3.0,
            b=self._init_std * 3.0,
        )

    def forward(self, x):
        # x is already sharded along in_features → local matmul is a partial sum
        partial = x.matmul(self.weight.t())
        return g(partial)  # all-reduce now; identity in backward


# ---------------------------------------------------------------------------
# Consolidated (world-size-agnostic) checkpointing.
#
# A running TP model is scattered: rank r holds only a SLICE of each sharded
# weight. We want checkpoints that DON'T bake in the training world size, so
# training and inference are fully decoupled — train on 8 GPUs, infer on 2 (or
# 1). Two operations make that work:
#
#   gather_full_state_dict(model)  -> reconstruct the FULL matrices (all_gather
#                                     + concat) so rank 0 can save ONE file that
#                                     is byte-identical to a single-GPU model.
#   load_full_state_dict(model, sd) -> each rank SLICES its own shard back out of
#                                     the full tensors (reshard on load), at
#                                     whatever the current world size is.
#
# Both rely on each parallel layer's `shard_dim` (0 for column, 1 for row) and
# the per-param `tp_sharded` flag. Everything else (embeddings, norms, output
# layer, buffers) is replicated and copied through unchanged.
# ---------------------------------------------------------------------------


def _sharded_weight_dims(model):
    """Map state_dict key -> shard_dim for every sharded weight in the model.

    A parallel layer named e.g. "transformer_blocks.0.gqa.W_Q" owns the
    state_dict key "transformer_blocks.0.gqa.W_Q.weight"; we tag that key with
    the layer's shard_dim so gather/slice know which axis to act on."""
    dims = {}
    for name, module in model.named_modules():
        if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
            dims[f"{name}.weight"] = module.shard_dim
    return dims


def _all_gather_cat(local, dim):
    """all_gather a per-rank shard and concatenate along `dim` → full tensor.

    Collective: EVERY rank must call this together. Each rank's shard has the
    same shape (dims divide evenly), which is what all_gather requires."""
    world = tp_world()
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local.contiguous())
    return torch.cat(gathered, dim=dim)


def gather_full_state_dict(model):
    """Reconstruct the full (unsharded) model state_dict.

    Returns the assembled dict ON RANK 0, and None on other ranks. Note every
    rank still PARTICIPATES in the all_gathers (they are collectives) — do not
    guard the body behind `if rank == 0` or it will deadlock. Tensors are moved
    to CPU so the saved file is portable across devices."""
    shard_dims = _sharded_weight_dims(model)
    local_sd = model.state_dict()
    is_rank0 = tp_rank() == 0
    full = {} if is_rank0 else None

    for key, tensor in local_sd.items():
        if key in shard_dims:
            # sharded weight → gather the slices from all ranks into the full matrix
            full_tensor = _all_gather_cat(tensor, shard_dims[key])
            if is_rank0:
                full[key] = full_tensor.detach().cpu()
        else:
            # replicated param/buffer → identical on every rank; rank 0 keeps its copy
            if is_rank0:
                full[key] = tensor.detach().cpu()
    return full


def load_full_state_dict(model, full_state_dict):
    """Load a full (unsharded) state_dict into this rank's TP model by SLICING
    each sharded weight down to this rank's shard. Replicated params/buffers are
    copied as-is. Works at ANY world size (including 1), regardless of the world
    size used when the file was saved — the file carries no sharding info."""
    shard_dims = _sharded_weight_dims(model)
    world = tp_world()
    rank = tp_rank()

    resharded = {}
    for key, tensor in full_state_dict.items():
        if key in shard_dims:
            dim = shard_dims[key]
            assert tensor.size(dim) % world == 0, (
                f"{key}: full size {tensor.size(dim)} on dim {dim} does not "
                f"divide world size {world}"
            )
            chunk = tensor.size(dim) // world
            # this rank's contiguous slice along the shard axis
            resharded[key] = tensor.narrow(dim, rank * chunk, chunk).clone()
        else:
            resharded[key] = tensor
    # shapes now match this rank's (sharded) parameters exactly
    model.load_state_dict(resharded)
