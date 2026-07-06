import torch
import torch.nn as nn
from src.blocks.rope import RoPE
from src.blocks.rmsnorm import RMSNorm
from src.distributed.parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    tp_world,
)


class GQA(nn.Module):
    """
    Grouped Query Attention layer
    """

    def __init__(
        self,
        context_length: int,
        hidden_dim: int,
        gka_ratio: int = 2,
        num_heads: int = 16,
    ):
        """
        Args:
            context_length (int): context length
            hidden_dim (int): hidden dimension
            gka_ratio (int, optional): ratio of grouped query attention. Defaults to 2.
            num_heads (int, optional): number of heads. Defaults to 16.
        """
        super().__init__()
        # Q/K/V are split by their OUTPUT dimension → each rank gets a subset of
        # the heads. No comm in the projection itself (f handles backward).
        self.W_Q = ColumnParallelLinear(hidden_dim, hidden_dim)  # split query heads
        self.W_K = ColumnParallelLinear(
            hidden_dim, int(hidden_dim // gka_ratio)
        )  # split KV heads
        self.W_V = ColumnParallelLinear(
            hidden_dim, int(hidden_dim // gka_ratio)
        )  # split KV heads
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.gka_ratio = gka_ratio
        # Output projection consumes the sharded heads → row-parallel; its g
        # operator all-reduces the partial outputs back to full hidden_dim.
        self.linear_output_layer = RowParallelLinear(hidden_dim, hidden_dim)

        # Per-rank head counts. Every reshape in forward must use THESE, not the
        # global counts, because each rank physically holds only its shard.
        world = tp_world()
        self.local_num_heads = num_heads // world
        self.local_num_kv_heads = (num_heads // gka_ratio) // world
        self.rope = RoPE(self.head_dim, context_length)
        self.rmsnorm = RMSNorm(self.head_dim)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(
        self, x: torch.tensor, token_positions: torch.tensor = None
    ) -> torch.tensor:
        """
        Args:
            x (torch.tensor): inputs of shape (batch_size, context_length, hidden_dim)
            token_positions (torch.tensor , optional): optional tensor with the positions of the tokens. Defaults to None.
        Returns:
            torch.tensor: output tensor of shape (batch_size, context_length, hidden_dim)
        """
        batch_size, context_length, hidden_dim = (
            x.shape
        )  # batch_size x context_length x 1024

        # compute Q, K, V matrices
        queries = self.W_Q(
            x
        )  # my_linear.forward(x) -> tensor: batch_size x context_length x 1024
        keys = self.W_K(
            x
        )  # my_linear.forward(x) -> tensor: batch_size x context_length x 512
        values = self.W_V(
            x
        )  # my_linear.forward(x) -> tensor: batch_size x context_length x 512

        # split into heads
        queries = queries.view(
            batch_size, context_length, self.head_dim, self.local_num_heads
        )  # local query heads only
        keys = keys.view(
            batch_size, context_length, self.head_dim, self.local_num_kv_heads
        )  # local KV heads only
        values = values.view(
            batch_size, context_length, self.head_dim, self.local_num_kv_heads
        )  # local KV heads only

        # normalize QK (CHECK THIS)
        queries = self.rmsnorm.forward(queries.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )  # batch_size x context_length x 64 x 16, normalize across hidden dimensions
        keys = self.rmsnorm.forward(keys.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )  # batch_size x context_length x 64 x 8, normalize across hidden dimensions

        # Compute RoPE embeddings
        token_positions = (
            torch.arange(context_length, device=x.device)
            if token_positions is None
            else token_positions
        )
        queries = self.rope.forward(
            queries, token_positions=token_positions
        )  # batch_size x context_length x 64 x 16
        keys = self.rope.forward(
            keys, token_positions=token_positions
        )  # batch_size x context_length x 64 x 8

        # reshape for attention computation
        queries = queries.view(
            batch_size,
            context_length,
            self.head_dim,
            self.local_num_kv_heads,
            self.gka_ratio,
        )  # split local query heads into (kv_groups, gka_ratio)
        keys = keys.unsqueeze(-1)  # batch_size x context_length x 64 x 8 x 1
        keys = keys.transpose(1, 2)  # batch_size x 64 x context_length x 8 x 1
        queries = queries.permute(
            0, 3, 4, 1, 2
        )  # [batch_size, 8,2, context_length, 64]
        keys = keys.permute(0, 3, 4, 1, 2)  # batch_size x 8 x 1 x 64 x context_length

        # compute attention scores
        attn_scores = (
            queries @ keys
        )  # batch_size x 8 x 2 x context_length x context_length
        mask = self.mask[:context_length, :context_length].bool()
        attn_scores = attn_scores.masked_fill(mask, -float("inf"))

        # compute attention weights
        attn_weights = torch.softmax(
            attn_scores / (self.head_dim**0.5), dim=-1
        )  # batch_size x 8 x 2 x context_length x context_length

        # compute context vector
        values = values.unsqueeze(-1)  # batch_size x context_length x 64 x 8 x 1
        values = values.permute(
            0, 3, 4, 1, 2
        )  # batch_size x 8 x 1 x context_length x 64
        context_vector = (
            attn_weights @ values
        )  # batch_size x 8 x 2 x context_length x 64
        context_vector = context_vector.view(
            batch_size, self.local_num_heads, context_length, self.head_dim
        )  # batch_size x local_num_heads x context_length x 64
        context_vector = context_vector.permute(
            0, 2, 1, 3
        )  # batch_size x context_length x 16 x 64

        # concatenate heads → this is the SHARDED hidden dim (local heads only),
        # exactly what RowParallelLinear expects as its sharded input.
        context_vector = context_vector.reshape(
            batch_size, context_length, self.local_num_heads * self.head_dim
        )  # batch_size x context_length x (local_num_heads * 64)

        # row-parallel output projection: local matmul → partial, then g
        # all-reduces across ranks back to the full hidden_dim.
        context_vector = self.linear_output_layer(
            context_vector
        )  # batch_size x context_length x 1024

        return context_vector
