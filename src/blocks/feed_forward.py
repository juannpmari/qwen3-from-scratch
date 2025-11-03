import torch
import torch.nn as nn

class SwigluFeedForward(nn.Module):
    
    def __init__(self, hidden_dim: int, dff: int, device: torch.device | None = None):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, dff, bias=False)
        self.W2 = nn.Linear(dff, hidden_dim, bias=False)
        self.W3 = nn.Linear(hidden_dim, dff, bias=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        w1 = self.W1(x)
        silu = w1 * torch.sigmoid(w1)
        return self.W2(silu * self.W3(x))
        

