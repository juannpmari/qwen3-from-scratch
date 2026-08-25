import os
import torch

from src.distributed.parallel import (
    tp_rank,
    gather_full_state_dict,
    load_full_state_dict,
)


def save_consolidated_checkpoint(model, iteration, out):
    """Save ONE world-size-agnostic checkpoint (weights only).

    Every rank participates in the gather (it is a collective); only rank 0
    reconstructs the full matrices and writes the file. The resulting
    model_state_dict is identical to a single-GPU model's, so it can later be
    loaded at ANY world size via load_consolidated_checkpoint.

    KNOWN LIMITATION: optimizer state is NOT saved here. AdamW's moment buffers
    shard exactly like their params, so consolidating them is possible but adds
    real complexity; we keep this file weights-only. Consequence: you can resume
    INFERENCE at any world size, but mid-training resume (which needs optimizer
    moments) across a changed world size is not supported by this file."""
    # collective — must run on ALL ranks, before the rank-0-only write below
    full_state_dict = gather_full_state_dict(model)
    if tp_rank() == 0:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save(
            {"iteration": iteration, "model_state_dict": full_state_dict},
            out,
        )


def load_consolidated_checkpoint(src, model):
    """Load a consolidated (full) checkpoint and reshard it into this rank's TP
    model. Each rank reads the same file and slices out its own shard, so this
    works even if the current world size differs from the one used to train.
    Returns the saved iteration number.

    mmap=True memory-maps the tensor storages instead of eagerly reading them,
    so when load_full_state_dict does `tensor.narrow(...).clone()` on a SHARDED
    weight, only that rank's shard is paged in from disk. Peak resident memory
    per rank drops from "the full model" to roughly "replicated params + this
    rank's shards" — avoiding every rank materializing the whole model.

    LIMITATION / FUTURE WORK: this only streams the SHARDED weights. Replicated
    params — which in this model include the embedding and LM head, the two
    biggest matrices — are read in full on every rank (they are needed in full).
    Truly memory-optimal loading requires (1) vocab-sharding those layers so they
    become sharded too, and (2) a sliceable on-disk format (e.g. safetensors
    get_slice) that reads only a shard's bytes rather than mmapping the full
    tensor. See docs/distributed.md Roadmap."""
    data = torch.load(src, map_location="cpu", mmap=True)
    load_full_state_dict(model, data["model_state_dict"])
    return data["iteration"]


def save_checkpoint(model, optimizer, iteration, out):
    """
    Save a checkpoint of the model and optimizer.
    Args:
        model: model to save
        optimizer: optimizer to save
        iteration: iteration number
        out: path to save the checkpoint
    """
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        out,
    )


def load_checkpoint(src, model, optimizer):
    """
    Load a checkpoint of the model and optimizer.
    Args:
        src: path to the checkpoint
        model: model to load the checkpoint into
        optimizer: optimizer to load the checkpoint into
    Returns:
        iteration: iteration number
    """
    data = torch.load(src)
    model.load_state_dict(data["model_state_dict"])
    optimizer.load_state_dict(data["optimizer_state_dict"])
    return data["iteration"]


# Test: uv run pytest -k test_checkpointing
