import torch.nn as nn
from rmsnorm import RMSNorm
from rope import Rope

class GQA(nn.Module):
    def __init__(self, emb_dim, gka_ratio = 2, num_heads = 16):
        super().__init__()
        self.W_Q = nn.Linear(emb_dim, emb_dim, bias=False) # 1024x1024 #no QKV bias in qwen3
        self.W_K = nn.Linear(emb_dim, emb_dim/gka_ratio, bias=False) # 1024x512 #no QKV bias in qwen3
        self.W_V = nn.Linear(emb_dim, emb_dim/gka_ratio, bias=False) # 1024x512 #no QKV bias in qwen3
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.gka_ratio = gka_ratio
        self.linear_output_layer = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        #compute Q, K, V marices
        queries = self.W_Q(x) # my_linear.forward(x) -> tensor: seq_len x 1024
        keys = self.W_K(x) # my_linear.forward(x) -> tensor: seq_len x 512
        values = self.W_V(x) # my_linear.forward(x) -> tensor: seq_len x 512
        
        #split into heads
        queries = queries.view(batch_size, seq_len, self.head_dim, self.num_heads) # batch_size x seq_len x 64 x 16
        keys = keys.view(batch_size, seq_len, self.head_dim, self.num_heads//self.gka_ratio) # batch_size x seq_len x 64 x 8
        values = values.view(batch_size, seq_len, self.head_dim, self.num_heads//self.gka_ratio) # batch_size x seq_len x 64 x 8
        
        #normalize QK (CHECK THIS)
        queries = RMSNorm(self.head_dim)(queries) # batch_size x seq_len x 64 x 16
        keys = RMSNorm(self.head_dim)(keys) # batch_size x seq_len x 64 x 8

        #Compute RoPE embeddings (CHECK THIS)
        queries = Rope(queries) # batch_size x seq_len x 64 x 16
        keys = Rope(keys) # batch_size x seq_len x 64 x 8

        
        #reshape for attention computation
        queries.view(batch_size, seq_len, self.head_dim, self.num_heads//self.gka_ratio, self.gka_ratio) # batch_size x seq_len x 64 x 8 x 2
        keys.view(batch_size, seq_len, self.head_dim, self.num_heads//self.gka_ratio,1) # batch_size x seq_len x 64 x 8 x 1

        #compute attention scores (CHECK THIS)
        attn_scores = queries*keys #element-wise multiplication with broadcasting
        
        #mask?

        #compute context vector
        context_vector = attn_scores*values

        #concatenate heads

        #apply linear output layer

        return context_vector
        

