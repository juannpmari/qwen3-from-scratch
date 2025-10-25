import torch

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.epsilon = eps
        self.gain = torch.nn.Parameter(torch.ones(hidden_dim))
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon)
        return self.gain * x/rms