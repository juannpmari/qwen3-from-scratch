import numpy as np
import torch

import torch
import numpy as np
from typing import Tuple

## Works but done by chatgpt
# def sample_data(x: np.ndarray, b: int, context_length: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Returns:
#         samples: (batch_size, context_length)
#         targets: (batch_size, context_length)
#     """
#     # 1D tensor of tokens
#     x_t = torch.from_numpy(x).to(device)

#     # number of valid starting positions is len(x) - context_length
#     # we sample starts in [0, len(x) - context_length - 1], so high is len(x) - context_length (exclusive)
#     max_start_exclusive = len(x) - context_length
#     if max_start_exclusive <= 0:
#         raise ValueError("context_length must be smaller than dataset length")

#     # sample random start indices (shape: (b,))
#     starts = torch.randint(0, max_start_exclusive, (b,), device=device)

#     # build indices for each sample: (b, context_length)
#     offsets = torch.arange(context_length, device=device).unsqueeze(0)  # (1, context_length)
#     idx = starts.unsqueeze(1) + offsets  # (b, context_length)

#     # gather samples and targets
#     samples = x_t[idx]                 # x[idx] indexes the 1D tensor x_t at each index -> shape (b, context_length)
#     targets = x_t[idx + 1]            # next-token targets

#     return samples, targets

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