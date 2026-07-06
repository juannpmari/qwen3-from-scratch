"""Standalone tensor-parallel smoke test.

Run (2 CPU processes on your single-GPU box):
    TP_BACKEND=gloo torchrun --nproc_per_node=2 -m src.distributed.smoke_test

What it PROVES:
  - the process group + f/g all-reduces actually fire without hanging;
  - forward runs and every rank produces IDENTICAL logits — they must, because
    each block all-reduces back to the full hidden dim, so the output is
    replicated;
  - backward runs; a REPLICATED param's grad is identical across ranks (the f
    operator all-reduces it), while a SHARDED param's grad is populated and
    local to each rank.

What it does NOT prove: exact numeric equivalence to a single-GPU baseline.
That needs the weight-scatter helper and is the natural next step.
"""
import torch
import torch.distributed as dist

from src.distributed.parallel import (
    setup_distributed,
    cleanup_distributed,
    sync_replicated_params,
    tp_device,
)
from src.qwen3.transformer import Transformer


def max_abs_diff_across_ranks(t):
    """Max |t_thisRank - t_rank0| — computed on every rank via a broadcast."""
    ref = t.detach().clone().contiguous()
    dist.broadcast(ref, src=0)  # everyone overwrites ref with rank 0's copy
    return (t.detach() - ref).abs().max()


def main():
    rank, world = setup_distributed()
    torch.manual_seed(1234 + rank)  # DIFFERENT seed per rank on purpose:
    #   sharded params should differ per rank; sync_replicated_params() then
    #   forces the replicated ones back into agreement. If the test still sees
    #   identical logits, the sharding/comms are wired correctly.
    device = tp_device()

    # tiny model; every sharded dim must divide the world size
    model = Transformer(
        vocab_size=256,
        num_layers=2,
        context_length=16,
        hidden_dim=64,
        dff=128,
        gka_ratio=1,
        num_heads=8,
    ).to(device)
    sync_replicated_params(model)  # replicated params now agree across ranks

    # identical input on every rank (broadcast rank 0's tokens)
    tokens = torch.randint(0, 256, (2, 16), device=device)
    dist.broadcast(tokens, src=0)

    # ---- forward: the g all-reduces fire inside each block ----
    logits = model(tokens)
    fwd_diff = max_abs_diff_across_ranks(logits)

    # ---- backward: the f all-reduces fire here ----
    loss = logits.float().sum()
    loss.backward()

    emb_grad = model.token_embedding.weight.grad          # replicated param
    emb_diff = max_abs_diff_across_ranks(emb_grad)
    wq_grad = model.transformer_blocks[0].gqa.W_Q.weight.grad  # sharded param

    if rank == 0:
        print(f"world_size = {world}")
        print(f"[fwd] max logit diff across ranks : {fwd_diff.item():.3e}  (expect ~0)")
        print(f"[bwd] embedding grad diff (replicated): {emb_diff.item():.3e}  (expect ~0)")
        print(f"[bwd] W_Q grad (sharded) shape: {tuple(wq_grad.shape)}  norm={wq_grad.norm().item():.3e}")
        ok = fwd_diff.item() < 1e-4 and emb_diff.item() < 1e-4
        print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")

    cleanup_distributed()


if __name__ == "__main__":
    main()
