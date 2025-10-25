import torch

class RoPE(nn.Module):
    def __init__(self, d: int, m: int, device=None):
        """
        d: head_dim
        m = context_length 
        device: torch.device | None = None - Device to store the buffer on
        """
        super().__init__()

        theta = torch.tensor([10000**(-2*i/d) for i in range(d//2)], dtype=torch.float32)
        sin_tensor = torch.sin(torch.outer(torch.arange(m, dtype=torch.float32), theta)) # m x d//2
        cos_tensor = torch.cos(torch.outer(torch.arange(m, dtype=torch.float32), theta)) # m x d//2
        
        self.register_buffer("sin_tensor", sin_tensor.to(device))
        self.register_buffer("cos_tensor", cos_tensor.to(device))

    def forward(self, x: torch.tensor): #check token_position
        """
        x: q or k of shape (batch_size, seq_len, head_dim, num_heads)
        returns: q or k of shape (batch_size, seq_len, head_dim, num_heads)
        """
        seq_len = x.shape[1]
        x = x.permute(0,3,1,2)
        first_half, second_half = x.chunk(2, dim=-1) #batch_size x num_heads x seq_len x head_dim//2

        first_half_rot = first_half * self.cos_tensor[:seq_len] - second_half * self.sin_tensor[:seq_len]
        second_half_rot = first_half * self.sin_tensor[:seq_len] + second_half * self.cos_tensor[:seq_len]

        concat = torch.cat([first_half_rot, second_half_rot], dim=-1)

        return concat.permute(0,2,3,1)


