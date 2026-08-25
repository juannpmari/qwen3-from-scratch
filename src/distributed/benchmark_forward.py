"""Single forward-pass timing benchmark under tensor parallelism.

A single forward has NO sampling, so unlike `generate` it is correct on every
rank (all ranks feed the same input → the all-reduces combine matching shards)
and needs none of the inference config plumbing. This is the minimal way to
measure TP forward latency.

Run (2 CPU processes here; DROP TP_BACKEND on a real multi-GPU box to use NCCL):
    TP_BACKEND=gloo torchrun --nproc_per_node=2 -m src.distributed.benchmark_forward

Real 2-GPU:
    torchrun --nproc_per_node=2 -m src.distributed.benchmark_forward

Optional env knobs:
    BENCH_BATCH    (default 1)
    BENCH_SEQ      (default = context_length from config)
    BENCH_ITERS    (default 20)   timed iterations
    BENCH_WARMUP   (default 5)    untimed warmup iterations
    BENCH_WEIGHTS  (default "" → random init; set to a consolidated ckpt path to
                    exercise the load+reshard path — weights don't affect timing)
"""
import os
import time
import statistics
import yaml

import torch
import torch.distributed as dist

from src.distributed.parallel import (
    setup_distributed,
    cleanup_distributed,
    seed_model_init,
    finalize_model_init,
    tp_rank,
    tp_device,
)
from src.qwen3.transformer import Transformer
from src.train.checkpointing import load_consolidated_checkpoint


def main():
    rank, world = setup_distributed()

    # model dims come straight from the training config so we benchmark the real
    # architecture (only the model + context_length fields are needed).
    with open("experiments/base_train.yaml") as fh:
        cfg = yaml.safe_load(fh)
    m = cfg["model"]
    ctx = cfg["training"]["context_length"]

    batch = int(os.environ.get("BENCH_BATCH", 1))
    seq = int(os.environ.get("BENCH_SEQ", ctx))
    iters = int(os.environ.get("BENCH_ITERS", 20))
    warmup = int(os.environ.get("BENCH_WARMUP", 5))
    weights = os.environ.get("BENCH_WEIGHTS", "")
    assert seq <= ctx, f"BENCH_SEQ={seq} exceeds context_length={ctx}"

    device = tp_device()

    # dual_rng gives a consistent, valid TP model even with no checkpoint.
    seed_model_init("dual_rng", 0)
    model = Transformer(
        vocab_size=m["vocab_size"],
        num_layers=m["num_layers"],
        context_length=ctx,
        hidden_dim=m["d_model"],
        dff=m["dff"],
        gka_ratio=m["gka_ratio"],
        num_heads=m["num_heads"],
    ).to(device)
    finalize_model_init(model, "dual_rng", 0)
    if weights:
        load_consolidated_checkpoint(weights, model)  # reshards into this rank
    model.eval()

    # identical dummy input on EVERY rank — TP requires all ranks to process the
    # same batch (broadcast rank 0's tokens to be certain they match).
    tokens = torch.randint(0, m["vocab_size"], (batch, seq), device=device)
    dist.broadcast(tokens, src=0)

    is_cuda = device.type == "cuda"

    def one_forward():
        with torch.no_grad():
            model(tokens)
        if is_cuda:
            torch.cuda.synchronize()  # kernels are async — wait for real completion

    for _ in range(warmup):
        one_forward()

    times_ms = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_forward()
        times_ms.append((time.perf_counter() - t0) * 1e3)

    if rank == 0:
        times_ms.sort()
        print(f"world_size={world}  backend={dist.get_backend()}  device={device}")
        print(
            f"model: layers={m['num_layers']} d_model={m['d_model']} "
            f"heads={m['num_heads']} dff={m['dff']} vocab={m['vocab_size']}"
        )
        print(f"input: batch={batch} seq={seq}")
        print(
            f"forward ms  mean={statistics.mean(times_ms):.2f}  "
            f"median={statistics.median(times_ms):.2f}  min={times_ms[0]:.2f}  "
            f"(warmup={warmup}, iters={iters})"
        )

    cleanup_distributed()


if __name__ == "__main__":
    main()
