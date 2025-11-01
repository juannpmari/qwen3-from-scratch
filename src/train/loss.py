import torch

def compute_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (batch_size, vocab_size) CHECK
    targets: (batch_size,) IDs of the target tokens

    Returns:
        ce: (batch_size,) cross entropy for each sequence in the batch
    """
    log_probs = torch.log_softmax(logits, dim=2)
    ce = -log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)
    return ce  # (batch_size, context_length)


    #uv run pytest -k test_cross_entropy