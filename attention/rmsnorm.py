import torch

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.epsilon = 1e-5
        self.gain = torch.nn.Parameter(torch.ones(hidden_dim))
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon)
        return self.gain * x/rms