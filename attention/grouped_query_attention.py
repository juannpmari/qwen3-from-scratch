import torch
import torch.nn as nn
from attention.rmsnorm import RMSNorm
from attention.rope import compute_rope

class GQA(nn.Module):
    def __init__(self, emb_dim, gka_ratio = 2, num_heads = 16):
        super().__init__()
        self.W_Q = nn.Linear(emb_dim, emb_dim, bias=False) # 1024x1024 #no QKV bias in qwen3
        self.W_K = nn.Linear(emb_dim, emb_dim//gka_ratio, bias=False) # 1024x512 #no QKV bias in qwen3
        self.W_V = nn.Linear(emb_dim, emb_dim//gka_ratio, bias=False) # 1024x512 #no QKV bias in qwen3
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.gka_ratio = gka_ratio
        self.linear_output_layer = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, x):
        """
        x: inputs of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape # batch_size x seq_len x 1024

        #compute Q, K, V marices
        queries = self.W_Q(x) # my_linear.forward(x) -> tensor: batch_size x seq_len x 1024
        keys = self.W_K(x) # my_linear.forward(x) -> tensor: batch_size x seq_len x 512
        values = self.W_V(x) # my_linear.forward(x) -> tensor: batch_size x seq_len x 512
        
        #split into heads
        queries = queries.view(batch_size, seq_len, self.head_dim, self.num_heads) # batch_size x seq_len x 64 x 16
        keys = keys.view(batch_size, seq_len, self.head_dim, self.num_heads//self.gka_ratio) # batch_size x seq_len x 64 x 8
        values = values.view(batch_size, seq_len, self.head_dim, self.num_heads//self.gka_ratio) # batch_size x seq_len x 64 x 8
        
        # #normalize QK (CHECK THIS)
        # queries = RMSNorm(self.head_dim)(queries) # batch_size x seq_len x 64 x 16
        # keys = RMSNorm(self.head_dim)(keys) # batch_size x seq_len x 64 x 8

        #Compute RoPE embeddings
        queries = compute_rope(queries) # batch_size x seq_len x 64 x 16
        keys = compute_rope(keys) # batch_size x seq_len x 64 x 8

        
        #reshape for attention computation
        queries = queries.view(batch_size, seq_len, self.head_dim, self.num_heads//self.gka_ratio, self.gka_ratio) # batch_size x seq_len x 64 x 8 x 2
        keys = keys.unsqueeze(-1) # batch_size x seq_len x 64 x 8 x 1
        keys = keys.transpose(1,2) # batch_size x 64 x seq_len x 8 x 1
        queries = queries.permute(0, 3, 4, 1, 2)  # [batch_size, 8,2, seq_len, 64]
        keys = keys.permute(0, 3, 4, 1, 2) # batch_size x 8 x 1 x 64 x seq_len

        #compute attention scores
        attn_scores = queries @ keys #batch_size x 8 x 2 x seq_len x seq_len
                        
        #mask?

        #compute attention weights
        attn_weights = torch.softmax(attn_scores/hidden_dim**0.5, dim=-1) # batch_size x 8 x 2 x seq_len x seq_len

        #compute context vector
        values =values.unsqueeze(-1) # batch_size x seq_len x 64 x 8 x 1
        values = values.permute(0,3,4,1,2) # batch_size x 8 x 1 x seq_len x 64
        context_vector = attn_weights @ values # batch_size x 8 x 2 x seq_len x 64
        context_vector = context_vector.view(batch_size, self.num_heads, seq_len, hidden_dim//self.num_heads) # batch_size x 16 x seq_len x 64
        context_vector = context_vector.permute(0,2,1,3) # batch_size x seq_len x 16 x 64
        
        #concatenate heads
        context_vector = context_vector.reshape(batch_size, seq_len, hidden_dim) # batch_size x seq_len x 1024

        #apply linear output layer
        context_vector = self.linear_output_layer(context_vector) # batch_size x seq_len x 1024
        
 
        return context_vector
        

