import torch

def compute_rope(x: torch.tensor) -> torch.tensor:
    """
    x: q or k of shape (batch_size, seq_len, head_dim)
    returns: q or k of shape (batch_size, seq_len, head_dim)
    """
    d = x.shape[-2]
    m = x.shape[1] 
    theta = torch.tensor([10000**(-2*i/d) for i in range(d//2)], dtype=torch.float32)

    sin_tensor = torch.sin(torch.outer(torch.arange(m, dtype=torch.float32), theta)) # m x d//2
    cos_tensor = torch.cos(torch.outer(torch.arange(m, dtype=torch.float32), theta)) # m x d//2

    x = x.permute(0,3,1,2)
    first_half, second_half = x.chunk(2, dim=-1)

    first_half_rot = first_half * cos_tensor - second_half * sin_tensor
    second_half_rot = first_half * sin_tensor + second_half * cos_tensor

    concat = torch.cat([first_half_rot, second_half_rot], dim=-1)

    return concat.permute(0,2,3,1)

    