import torch


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
