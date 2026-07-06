# Tensor Parallelism

This document describes the tensor-parallel (TP) implementation in this repo. It
is an **educational** implementation: the priority is clarity over performance.
The whole scheme follows the Megatron-LM pattern and lives almost entirely in
`src/distributed/parallel.py`, with small, surgical changes to the attention and
MLP blocks.

> **Verification status.** Everything below has been exercised with **2 CPU
> processes over the `gloo` backend** via the smoke test. It has **not** yet been
> run on real multi-GPU / NCCL hardware — the GPU path is the same code with an
> auto-selected backend and device, but that seam is untested. Claims are written
> to reflect this.

---

## 1. Overview: what TP is (and the SPMD model)

Tensor parallelism does **not** split the model by layer (that is pipeline
parallelism). It splits *individual weight matrices* across ranks, so **every
rank runs every layer** but holds only a *slice* of each parallelized matrix.

Execution follows the **SPMD** model (Single Program, Multiple Data):
`torchrun --nproc_per_node=N main.py` launches **N identical processes**, each
running the *entire* program top to bottom. There is no central coordinator and
no automatic work partitioning. Consequently:

- **Every line runs on every rank** unless it is explicitly guarded with
  `if tp_rank() == 0:`.
- Ranks differ only through (a) rank-dependent values (`tp_rank()`, `tp_world()`
  drive which slice a layer keeps), (b) per-rank RNG state, and (c) **collectives**
  (`all_reduce`, `broadcast`) — the only points where ranks actually communicate.
- A collective is a *rendezvous*: **all** ranks must call it together, or the ones
  that did will hang forever waiting (deadlock).

Process-group lifecycle helpers (`src/distributed/parallel.py`):

- `setup_distributed()` — reads `RANK` / `LOCAL_RANK` / `WORLD_SIZE` (set by
  `torchrun`), picks a backend, calls `dist.init_process_group(...)`, returns
  `(rank, world_size)`. Must run **before** the model is built (the parallel
  layers read `tp_world()` in their constructor).
- `cleanup_distributed()` — `dist.destroy_process_group()` at shutdown.
- `tp_rank()`, `tp_world()` — thin wrappers over `dist.get_rank()` /
  `dist.get_world_size()`.

---

## 2. The `f` and `g` operators

All communication is funnelled through two tiny `torch.autograd.Function`s that
are conjugates of each other (`_F`/`_G` in `parallel.py`, wrapped by the
lowercase helpers `f(x)` / `g(x)`):

| Operator | Forward | Backward | Placed at |
|----------|---------|----------|-----------|
| `f` | identity | `all_reduce(SUM)` | **input** of a column-parallel region |
| `g` | `all_reduce(SUM)` | identity | **output** of a row-parallel region |

Why this works:

- `f` sits where the input `x` is **replicated** on every rank. Forward is a
  no-op. In backward, each rank produces a *partial* gradient w.r.t. that
  replicated input; summing them (`all_reduce`) yields the true gradient.
- `g` sits where each rank has produced a **partial** output (its slice of a sum).
  Forward all-reduces the partials into the full result. In backward the incoming
  gradient is already identical on every rank, so it passes straight through.

Because these are autograd Functions, we write the **forward** communication once
and autograd inserts the **backward** communication automatically — we never
hand-write a backward `all_reduce`. Both all-reduces use `dist.ReduceOp.SUM` and
operate in place (the tensors are made `.contiguous()` first).

---

## 3. `ColumnParallelLinear` and `RowParallelLinear`

These two modules replace the plain `Linear` from `src/common/linear.py`. Both
mirror its convention: `weight` has shape `(out_features, in_features)` and
forward is `x @ weight.t()`.

**`ColumnParallelLinear`** — splits the weight along its **output** dimension.
Each rank stores `out_features // world` rows and produces a *shard* of the output
columns.

```
forward:  x = f(x); return x @ weight_shard.t()   # replicated x → sharded output
```

No communication in forward; `f` provides the backward all-reduce. Asserts
`out_features % world == 0`.

**`RowParallelLinear`** — splits the weight along its **input** dimension. Each
rank stores `in_features // world` columns, consumes the matching input shard, and
produces a *partial* output.

```
forward:  partial = x_shard @ weight_shard.t(); return g(partial)  # sharded → replicated
```

`g` all-reduces the partials into the full output. Asserts
`in_features % world == 0`.

**The column→row pairing** is the key efficiency trick. If a column-parallel layer
feeds directly into a row-parallel layer, the intermediate activation stays
*sharded* the whole way through — no communication between them. You pay exactly
**one `f` (backward)** at the entry and **one `g` (forward)** at the exit of the
pair. This is why each attention block and each MLP block costs a single
all-reduce per direction.

Both layers keep their initialization variance based on the **full** dimensions
(`init_std = sqrt(2 / (in_features + out_features))`), matching what the original
single-GPU `Linear` would have used, and mark their weight with
`weight.tp_sharded = True` so the init/reconcile logic can tell shards apart from
replicated params. The init is factored into `reset_parameters()` so it can be
re-drawn later (see §6).

---

## 4. Where TP is applied

### Grouped-Query Attention — `src/blocks/grouped_query_attention.py`

- `W_Q`, `W_K`, `W_V` → `ColumnParallelLinear`. This splits the **heads** across
  ranks: each rank computes a subset of the query heads (and the matching KV
  heads).
- `linear_output_layer` → `RowParallelLinear`. It consumes the concatenated local
  heads (the sharded hidden dim) and its `g` all-reduces back to the full
  `hidden_dim`.

**The reshape subtlety.** After sharding, each rank physically holds fewer heads,
so every `.view()`/`.reshape()` that used the global head count must use the
**per-rank** count. The constructor computes:

```python
self.local_num_heads    = num_heads // world
self.local_num_kv_heads = (num_heads // gka_ratio) // world
```

and `forward` uses `self.local_num_heads` / `self.local_num_kv_heads` for the
head split, the query `(kv_groups, gka_ratio)` reshape, the context reshape, and
the final head-concat (`self.local_num_heads * self.head_dim`). Getting one of
these wrong silently corrupts the tensor layout, so this is the fiddliest part of
the whole change.

**RoPE and the QK-`RMSNorm` are untouched.** They operate *per head* on
`head_dim`, and `head_dim` is not sharded — only the *number of heads* is. Each
rank simply applies them to its own head subset, no communication required. The
causal `mask` buffer is likewise per-rank-identical.

### SwiGLU MLP — `src/blocks/feed_forward.py`

- `W1` and `W3` → `ColumnParallelLinear`. Both produce the `dff`-sized
  intermediate, so both are split along `dff`. The elementwise SiLU gate
  (`silu * W3(x)`) then runs on the local `dff` shard with no communication —
  crucially, **both** gate operands must be sharded the same way.
- `W2` → `RowParallelLinear`. It consumes the sharded `dff` and its `g`
  all-reduces back to full `hidden_dim`.

The MLP `forward` needed **no** code changes — the sharding is invisible to it.
Only the layer *types* changed.

### Residual fix — `src/qwen3/transformer.py`

`TransformerBlock.forward` had to switch its residual adds from in-place
`x += residual` to out-of-place `x = x + residual`. The output of `GQA`/`SwiGLU`
is the output of the `g` custom autograd Function, and **modifying a custom
Function's output in place is forbidden by PyTorch** — it would override our
hand-written backward and corrupt gradients. This was a real error surfaced by the
smoke test, not a hypothetical.

The rest of `Transformer` (token embedding, final RMSNorm, `output_layer`) is
**not** sharded — see the Roadmap for why.

---

## 5. Backends and devices

`setup_distributed()` selects the collective backend:

- **`nccl`** — real GPUs. Chosen automatically when `torch.cuda.is_available()`.
  Also calls `torch.cuda.set_device(local_rank)`.
- **`gloo`** — the CPU cousin of NCCL. Used to **emulate** multiple ranks as CPU
  processes on a single-GPU (or no-GPU) box.

The choice can be forced with the `TP_BACKEND=gloo|nccl` environment variable;
otherwise it auto-detects. Emulating with `gloo` faithfully reproduces the
*algorithm and collective semantics* — the same ring all-reduce, the same
`f`/`g` behaviour — but **not** performance, NVLink topology, or NCCL-specific
behaviour.

Three related helpers, easily confused:

| Helper | Returns | Scope | Used for |
|--------|---------|-------|----------|
| `tp_rank()` | `dist.get_rank()` | **global** across the whole job | choosing the leader (`== 0`) for logging/saving |
| `tp_local_rank()` | `LOCAL_RANK` env | **per-node** (`0..gpus_per_node-1`) | picking the physical GPU |
| `tp_device()` | `cuda:{local_rank}` or `cpu` | derived | where this rank's tensors live |

`tp_device()` is tied to the backend: `gloo → cpu`, `nccl → cuda:{local_rank}`.
On a single machine all three collapse (rank 0 → local_rank 0 → `cuda:0`); the
distinction only matters across multiple nodes.

---

## 6. Replicated vs. sharded parameters, and the two init strategies

Parameters fall into two classes:

- **Sharded** — the parallel-linear weights (`W_Q/K/V`, attention output proj,
  `W1/W2/W3`). Each rank holds a *different* slice; they are tagged
  `weight.tp_sharded = True`.
- **Replicated** — everything else (token embedding, all `RMSNorm`s,
  `output_layer`). Every rank must hold an *identical* copy.

Because each process constructs the model with its own RNG, replicated params
start out *different* across ranks and must be reconciled. There are two
strategies, selected by the `init_mode` config key (default **`dual_rng`**), and
implemented as a pair of calls that bracket model construction:

```python
seed_model_init(mode, shared_seed)            # BEFORE building
model = Transformer(...); model.to(device)
finalize_model_init(model, mode, shared_seed) # AFTER building
```

**`broadcast`** — every rank random-inits everything, then `sync_replicated_params`
broadcasts rank 0's replicated params to all ranks (sharded params are skipped via
the `tp_sharded` flag). Simple and explicit, but ranks `1..N-1` *waste* their
replicated-param init, which is immediately overwritten. (Note: only the *init
compute* is wasted — replicated params need the memory on every rank regardless.)

**`dual_rng`** (default) — `seed_model_init` sets a **shared** global seed so
every rank builds byte-identical params (replicated *and* sharded) with **no
communication**. Then `finalize_model_init` calls `_reinit_sharded_params`, which
temporarily seeds the RNG with a **per-rank** seed (`shared_seed + 1 + tp_rank()`)
and calls `reset_parameters()` on only the sharded layers, giving each rank a
distinct shard. The global RNG state is saved and restored around this so nothing
downstream is perturbed. No wasted init, no broadcast.

`broadcast` is kept as the clearer teaching example; `dual_rng` is the leaner
default. Both are verified by the smoke test.

---

## 7. Data loading and RNG

Under pure TP, **every rank must see the same batch** each step — the ranks
collaborate on a single forward pass, so mismatched inputs would make the
all-reduces combine unrelated activations (silently wrong, not a crash). The data
already lives in the same file on disk for every rank, so the *only* source of
divergence is randomness. There are **two** RNG sources in the pipeline
(`src/preprocessing/dataloader.py`):

1. **Random window sampling** — `sample_data` uses `torch.randint(...)` to pick
   the starting offsets of the training windows. This randomizes the dataset
   *content*, not just its order.
2. **DataLoader shuffle** — `shuffle=True` randomizes batch *order*.

Both are made deterministic by a single **local** `torch.Generator`, seeded
identically on every rank and threaded into `randint` (via `sample_data` /
`CustomDataset`) and into the `DataLoader(generator=...)`. Same seed → identical
windows *and* identical order → identical batches, with **zero communication**.
`load_data` (`src/train/utils.py`) supplies the seed (`data_seed`, default 0) —
the same value for `train`, `+1` for `val`.

**Why a local `Generator` and not `torch.manual_seed`.** A global
`torch.manual_seed` would *also* freeze the global RNG that the model's weight
init draws from — making the **sharded** params identical on every rank and
defeating TP entirely. A local generator surgically seeds only the data pipeline
and leaves the global RNG per-rank-distinct. This property (data deterministic,
global RNG untouched) was checked directly.

---

## 8. Training loop specifics — `src/train/train.py`

- **Rank-0-only I/O.** Per-step loss prints, the validation-loss print, and the
  checkpoint write are guarded by `if tp_rank() == 0:` so N processes don't
  clobber each other's output/files. Loss values are still appended to the
  tracking lists on every rank (harmless; they're identical anyway since logits
  are replicated).
- **Validation runs on ALL ranks.** The validation forward pass (`model(input_val)`)
  contains the same `g`/`f` all-reduces as training. If it were guarded to rank 0
  only, the other ranks would never reach those collectives and rank 0 would hang
  forever. Only the *print* and the *save* around it are rank-0-only. This is the
  classic collective-mismatch deadlock, deliberately avoided here.

---

## 9. Checkpointing

**Design goal: the checkpoint format is world-size-agnostic**, so training and
inference are fully decoupled — you can train on N ranks and generate on M
(including M = 1).

- **Save (gather / consolidate).** At checkpoint time, each *sharded* weight is
  `all_gather`ed across ranks and concatenated along its shard dimension
  (`ColumnParallelLinear` → dim 0 / output rows; `RowParallelLinear` → dim 1 /
  input columns). Replicated params are already identical, so rank 0 takes its
  own copy. Rank 0 then writes **one** file whose `state_dict` is identical to
  what a single-GPU model would have produced. The training world size leaves no
  trace in the file.
- **Load (reshard).** `generate` builds a TP model at *its own* world size, loads
  the consolidated full `state_dict`, and each rank slices out its own shard along
  the same shard dimension. Loaded weights are authoritative, so **no** re-init or
  reconcile runs afterward.

> **Status.** The consolidated save/reshard-load is being added in a concurrent
> change; the interim implementation saved one **per-rank** shard file
> (`checkpoint_{epoch}_rank{N}.pt`) so no shard was lost. The consolidated format
> above is the intended design and supersedes the per-rank files.

Known rough edges are listed in the Roadmap (optimizer-state handling, memory
during gather).

---

## 10. How to run

**Smoke test** (2 CPU processes, no GPU needed):

```bash
TP_BACKEND=gloo torchrun --nproc_per_node=2 -m src.distributed.smoke_test
```

It builds a tiny `Transformer`, runs forward + backward, and checks:

- **forward** — every rank produces **identical logits** (max diff ≈ 0), proving
  the `g` all-reduce correctly replicates the output;
- **backward, replicated** — the token-embedding gradient is **identical across
  ranks** (the `f` all-reduce did its job);
- **backward, sharded** — the `W_Q` gradient is **local** to each rank and has the
  sharded shape.

It can be run with either init mode via `TP_INIT_MODE=dual_rng|broadcast`. Both
modes have been verified to pass. The smoke test does **not** prove exact numeric
equivalence to a single-GPU baseline — that requires the weight-scatter helper and
is future work.

**Training:**

```bash
torchrun --nproc_per_node=N main.py    # backend/device auto-selected
```

`N` must divide every sharded dimension (`num_heads`, `num_kv_heads`, `dff`,
`hidden_dim`); a bad `N` trips a clear assertion in the parallel layers. Training
also requires the tokenized data to exist at the configured paths.

---

## 11. Benchmarking

*TODO — not yet measured.* (Wall-clock and memory scaling across ranks/backends
to be added here.)

---

## 12. Roadmap

### Done

- Megatron-style TP for the **attention and MLP blocks** (Q/K/V + output proj;
  W1/W2/W3).
- The `f`/`g` conjugate autograd operators (backward comms automatic).
- `ColumnParallelLinear` / `RowParallelLinear` replacing the plain `Linear`.
- Both replicated-param init strategies: `broadcast` and `dual_rng` (default),
  config-driven via `init_mode`.
- Backend auto-selection with `TP_BACKEND` override: `nccl` (GPU) and `gloo`
  (CPU emulation).
- Deterministic, communication-free data loading (shared-seed local generator
  covering both the window sampling and the shuffle).
- Rank-0-only logging/writes; all-rank validation forward (deadlock-safe).
- Checkpointing: per-rank shard files (interim) and consolidated world-size-
  agnostic save + reshard-on-load (in progress).
- **Verified on CPU / `gloo` with 2 processes** via the smoke test, in both init
  modes.

### Known limitations / not done

- **Embedding and final output (LM-head) are not sharded.** These are the two
  *largest* matrices in the model (`vocab × hidden`). They currently sit
  replicated on every rank, so TP here shards the middle of the network but not
  its fattest layers — meaning it is **not memory-optimal**. Sharding them
  properly requires a **vocab-parallel embedding** and a **vocab-parallel
  cross-entropy** loss (the loss itself must become distributed-aware: partial
  logits per rank, with small all-reduces for the softmax max and denominator).
- **Single node only.** Multi-node runs need launch-time rendezvous configuration
  (`--rdzv_endpoint`, `MASTER_ADDR`, etc.). In practice TP is usually kept
  *intra-node* anyway, because it does an all-reduce at every layer and wants the
  fast NVLink interconnect; scaling wider is normally done with data/pipeline
  parallelism across nodes, not more TP ranks.
- **Checkpoint streaming for very large models.** The consolidated save gathers
  full matrices onto rank 0, which must briefly hold them in memory. Truly large
  models would need a streaming/sharded-write scheme.
- **Optimizer-state handling in consolidated checkpoints.** Gathering/resharding
  the optimizer moments (not just the weights) is not yet addressed.
- **`generate` checkpoint loading.** The per-rank vs. consolidated loading path in
  inference still needs to be finalized to match the checkpoint format.
- **Not yet run on real multi-GPU / NCCL.** The GPU path is the same code with an
  auto-selected backend and device, but it has only been exercised on CPU/`gloo`.
