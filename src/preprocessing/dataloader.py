import numpy as np
import torch

def sample_data(x: np.ndarray, b: int, context_length: int, device: torch.device) -> (torch.Tensor, torch.Tensor):
    """
    Args:
        x (np.ndarray): integer array with token IDs to sample from
        b (int): batch size
        context_length (int): context length
        device (torch.device): device to use
    Returns:
        (torch.Tensor, torch.Tensor): the sampled input sequences and the corresponding next-token targets (each (batch_size,context_length))
    """

    x = torch.from_numpy(x).to(device)
    max_start_exclusive = len(x) - context_length
    
    starts = torch.randint(0, max_start_exclusive, (b,), device=device)
    offsets = torch.arange(context_length, device=device).unsqueeze(0)  # (1, context_length)
    idx = starts.unsqueeze(1) + offsets  # (b, context_length)

    samples = x[idx]
    targets = x[idx + 1]
    return samples, targets

#TODO: Fix and replace by yield to return a generator
# Investigate how to do this more efficiently
# Use np.memmap or the flag mmap_mode='r' to np.load when loading the dataset


#test: uv run pytest -k test_get_batch