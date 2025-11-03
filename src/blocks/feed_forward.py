import torch
import torch.nn as nn
from src.common.linear import Linear

class SwigluFeedForward(nn.Module):
    
    def __init__(self, hidden_dim: int, dff: int, device: torch.device | None = None):
        super().__init__()
        self.W1 = Linear(hidden_dim, dff, device=device)
        self.W2 = Linear(dff, hidden_dim, device=device)
        self.W3 = Linear(hidden_dim, dff, device=device)

    def forward(self, x: torch.tensor) -> torch.tensor:
        w1 = self.W1(x)
        silu = w1 * torch.sigmoid(w1)
        return self.W2(silu * self.W3(x))
        

