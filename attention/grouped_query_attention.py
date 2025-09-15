import torch.nn as nn

class GQA(nn.Module):
    def __init__(self, emb_dim, gka_ratio = 2, num_heads = 16):
        super().__init__()
        self.W_Q = nn.Linear(emb_dim, emb_dim, bias=False) # 1024x1024
        self.W_K = nn.Linear(emb_dim, emb_dim/gka_ratio, bias=False) # 1024x512
        self.W_V = nn.Linear(emb_dim, emb_dim/gka_ratio, bias=False) # 1024x512
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.gka_ratio = gka_ratio

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
        