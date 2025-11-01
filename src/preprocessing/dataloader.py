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
    samples = torch.zeros((b, context_length), dtype=torch.long, device=device)
    targets = torch.zeros((b, context_length), dtype=torch.long, device=device)
    x = torch.from_numpy(x).to(device)
    for i in range(len(x) - context_length):
        samples[i] = x[i:i+context_length]
        targets[i] = x[i+1:i+context_length+1]
    return samples, targets
    #TODO: Fix and replace by yield to return a generator
    # Use np.memmap or the flag mmap_mode='r' to np.load when loading the dataset

    #test: uv run pytest -k test_get_batch