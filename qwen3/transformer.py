from common.grouped_query_attention import GQA
from common.rmsnorm import RMSNorm
from common.feed_forward import SwigluFeedForward
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, context_length: int, dff: int, gka_ratio: int = 2, num_heads: int = 16, device: torch.device | None = None):
        """
        hidden_dim: hidden dimension of the model
        context_length: length of the context
        dff: dimension of the feed forward layer
        gka_ratio: ratio of the number of heads to the number of groups
        num_heads: number of heads
        device: device to run on
        """
        super().__init__()
        self.rmsnorm = RMSNorm(hidden_dim, device)
        self.gqa = GQA(context_length, hidden_dim, gka_ratio, num_heads, device)
        self.ff = SwigluFeedForward(hidden_dim, dff, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: inputs of shape (batch_size, seq_len, hidden_dim)
        """
        residual = x
        x = self.rmsnorm(x)
        x = self.gqa(x)
        x += residual
        residual = x
        x = self.rmsnorm(x)
        x = self.ff(x)
        x += residual
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, context_length: int, hidden_dim: int, dff: int, gka_ratio: int = 2, num_heads: int = 16, device: torch.device | None = None):
        """
        vocab_size: size of the vocabulary
        num_layers: number of layers
        hidden_dim: hidden dimension of the model
        context_length: length of the context
        dff: dimension of the feed forward layer
        gka_ratio: ratio of the number of heads to the number of groups
        num_heads: number of heads
        device: device to run on
        """
        super().__init__()
        self.transformer_blocks = nn.ModuleList([TransformerBlock(hidden_dim, context_length, dff, gka_ratio, num_heads, device) for _ in range(num_layers)])
    