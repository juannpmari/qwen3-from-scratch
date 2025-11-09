import torch
import torch.nn as nn
from src.common.linear import Linear


class SwigluFeedForward(nn.Module):
    """
    SwiGLU feed forward layer
    """

    def __init__(self, hidden_dim: int, dff: int, device: torch.device | None = None):
        """
        Args:
            hidden_dim (int): hidden dimension
            dff (int): dimension of the feed forward layer
            device (torch.device | None, optional): device to run on. Defaults to None.
        """
        super().__init__()
        self.W1 = Linear(hidden_dim, dff, device=device)
        self.W2 = Linear(dff, hidden_dim, device=device)
        self.W3 = Linear(hidden_dim, dff, device=device)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x (torch.tensor): input tensor
        Returns:
            torch.tensor: output tensor
        """
        w1 = self.W1(x)
        silu = w1 * torch.sigmoid(w1)
        return self.W2(silu * self.W3(x))
