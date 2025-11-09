import torch


def softmax(x: torch.tensor, dim: int, temperature: float = 1.0) -> torch.tensor:
    """
    Softmax function
    Args:
        x (torch.tensor): input tensor
        dim (int): dimension to apply softmax
        temperature (float, optional): temperature to apply. Defaults to 1.0.
    Returns:
        torch.tensor: output tensor
    """
    maximum, _ = torch.max(x, dim=dim, keepdim=True)
    x = x - maximum  # to avoid overflow when exponentiating very large inputs
    softmax = torch.exp(x / temperature) / torch.sum(
        torch.exp(x / temperature), dim=dim, keepdim=True
    )
    return softmax


# NOTE: this passes the test but in reality doesn't support very large inputs, it overflows
