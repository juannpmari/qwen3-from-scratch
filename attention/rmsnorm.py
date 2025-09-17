import torch

def compute_rmsnorm(x: torch.tensor) -> torch.tensor:
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
    return x / rms