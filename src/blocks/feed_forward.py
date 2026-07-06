import torch
import torch.nn as nn
from src.distributed.parallel import ColumnParallelLinear, RowParallelLinear


class SwigluFeedForward(nn.Module):
    """
    SwiGLU feed forward layer
    """

    def __init__(self, hidden_dim: int, dff: int, device: torch.device  = None):
        """
        Args:
            hidden_dim (int): hidden dimension
            dff (int): dimension of the feed forward layer
            device (torch.device , optional): device to run on. Defaults to None.
        """
        super().__init__()
        # W1 and W3 both produce the dff-sized intermediate → column-parallel
        # (split dff across ranks). The elementwise SiLU gate then runs on the
        # local dff shard with no communication.
        self.W1 = ColumnParallelLinear(hidden_dim, dff, device=device)
        # W2 consumes the sharded dff → row-parallel; its g operator all-reduces
        # the partial outputs back to the full hidden_dim.
        self.W2 = RowParallelLinear(dff, hidden_dim, device=device)
        self.W3 = ColumnParallelLinear(hidden_dim, dff, device=device)

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
