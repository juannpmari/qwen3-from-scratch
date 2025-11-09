import torch


def compute_perplexity(cross_entropy: torch.Tensor) -> torch.Tensor:
    """
    cross_entropy: (batch_size, context_length)

    Returns:
        perplexity: (batch_size)
    """
    return torch.exp(torch.mean(cross_entropy, dim=1))  # (batch_size)
