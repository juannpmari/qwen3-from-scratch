import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device: torch.device | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model, device=device))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.tensor) -> torch.tensor:
        return self.weight[token_ids]
        