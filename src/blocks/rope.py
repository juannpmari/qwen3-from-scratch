import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(self, d: int, m: int, device:torch.device = None, dtype:torch.dtype = None):
        """
        d: head_dim
        m = context_length 
        device: torch.device | None = None - Device to store the buffer on
        dtype: torch.dtype | None = None - Data type to store the buffer on
        """
        super().__init__()

        theta = torch.tensor([10000**(-2*i/d) for i in range(d//2)], device=device, dtype=dtype)
        sin_tensor = torch.sin(torch.outer(torch.arange(m), theta), device=device, dtype=dtype) # m x d//2
        cos_tensor = torch.cos(torch.outer(torch.arange(m), theta), device=device, dtype=dtype) # m x d//2
        
        self.register_buffer("sin_tensor", sin_tensor)
        self.register_buffer("cos_tensor", cos_tensor)

   
    def forward(self, x: torch.tensor, token_positions: torch.tensor): #check token_position
        """
        x: q or k of shape (batch_size, context_length, head_dim, num_heads)
        token_positions: (batch_size, context_length) Position of the tokens in the sequence
        returns: q or k of shape (batch_size, context_length, head_dim, num_heads)
        """
        

        # context_length = x.shape[1]
        x = x.permute(0,3,1,2)
        
        x_even = x[..., 0::2] #batch_size x num_heads x context_length x head_dim//2
        x_odd = x[..., 1::2] #batch_size x num_heads x context_length x head_dim//2

        x_even_rot = x_even * self.cos_tensor[token_positions] - x_odd * self.sin_tensor[token_positions]
        x_odd_rot = x_even * self.sin_tensor[token_positions] + x_odd * self.cos_tensor[token_positions]

        concat = torch.stack([x_even_rot, x_odd_rot], dim=-1).flatten(-2)
        result = concat.permute(0,2,3,1)

        return result


