import torch
import torch.nn as nn
from src.blocks.rope import RoPE
from src.blocks.rmsnorm import RMSNorm
from src.common.linear import Linear

class GQA(nn.Module):
    def __init__(self, context_length, hidden_dim, gka_ratio = 2, num_heads = 16, device = None):
        super().__init__()
        self.W_Q = Linear(hidden_dim, hidden_dim)#, device=device) # 1024x1024 #no QKV bias in qwen3
        self.W_K = Linear(hidden_dim, int(hidden_dim//gka_ratio))#, device=device) # 1024x512 #no QKV bias in qwen3
        self.W_V = Linear(hidden_dim, int(hidden_dim//gka_ratio))#, device=device) # 1024x512 #no QKV bias in qwen3
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.gka_ratio = gka_ratio
        self.linear_output_layer = Linear(hidden_dim, hidden_dim)#, device=device)
        self.rope = RoPE(self.head_dim, context_length)#, device)
        self.rmsnorm = RMSNorm(self.head_dim)#, device=device)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, token_positions = None):
        """
        x: inputs of shape (batch_size, context_length, hidden_dim)
        token_positions: optional tensor with the positions of the tokens
        """
        batch_size, context_length, hidden_dim = x.shape # batch_size x context_length x 1024

        #compute Q, K, V matrices
        queries = self.W_Q(x) # my_linear.forward(x) -> tensor: batch_size x context_length x 1024
        keys = self.W_K(x) # my_linear.forward(x) -> tensor: batch_size x context_length x 512
        values = self.W_V(x) # my_linear.forward(x) -> tensor: batch_size x context_length x 512

        #split into heads
        queries = queries.view(batch_size, context_length, self.head_dim, self.num_heads) # batch_size x context_length x 64 x 16
        keys = keys.view(batch_size, context_length, self.head_dim, self.num_heads//self.gka_ratio) # batch_size x context_length x 64 x 8
        values = values.view(batch_size, context_length, self.head_dim, self.num_heads//self.gka_ratio) # batch_size x context_length x 64 x 8

        #normalize QK (CHECK THIS) 
        queries = self.rmsnorm.forward(queries.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # batch_size x context_length x 64 x 16, normalize across hidden dimensions
        keys = self.rmsnorm.forward(keys.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # batch_size x context_length x 64 x 8, normalize across hidden dimensions

        #Compute RoPE embeddings
        token_positions = torch.arange(context_length) if token_positions is None else token_positions
        queries = self.rope.forward(queries, token_positions = token_positions) # batch_size x context_length x 64 x 16
        keys = self.rope.forward(keys, token_positions = token_positions) # batch_size x context_length x 64 x 8
        
        #reshape for attention computation
        queries = queries.view(batch_size, context_length, self.head_dim, int(self.num_heads//self.gka_ratio), self.gka_ratio) # batch_size x context_length x 64 x 8 x 2
        keys = keys.unsqueeze(-1) # batch_size x context_length x 64 x 8 x 1
        keys = keys.transpose(1,2) # batch_size x 64 x context_length x 8 x 1
        queries = queries.permute(0, 3, 4, 1, 2)  # [batch_size, 8,2, context_length, 64]
        keys = keys.permute(0, 3, 4, 1, 2) # batch_size x 8 x 1 x 64 x context_length

        #compute attention scores
        attn_scores = queries @ keys #batch_size x 8 x 2 x context_length x context_length
        mask = self.mask[:context_length, :context_length].bool()
        attn_scores = attn_scores.masked_fill(mask, -float('inf'))
                        
        #compute attention weights
        attn_weights = torch.softmax(attn_scores/(self.head_dim**0.5), dim=-1) # batch_size x 8 x 2 x context_length x context_length

        #compute context vector
        values = values.unsqueeze(-1) # batch_size x context_length x 64 x 8 x 1
        values = values.permute(0,3,4,1,2) # batch_size x 8 x 1 x context_length x 64
        context_vector = attn_weights @ values # batch_size x 8 x 2 x context_length x 64
        context_vector = context_vector.view(batch_size, self.num_heads, context_length, hidden_dim//self.num_heads) # batch_size x 16 x context_length x 64
        context_vector = context_vector.permute(0,2,1,3) # batch_size x context_length x 16 x 64
        
        #concatenate heads
        context_vector = context_vector.reshape(batch_size, context_length, hidden_dim) # batch_size x context_length x 1024

        #apply linear output layer
        context_vector = self.linear_output_layer(context_vector) # batch_size x context_length x 1024
        
        return context_vector
        

