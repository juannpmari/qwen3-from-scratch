import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocessing.tokenizer import BPETokenizer
from typing import List
import tiktoken


def sample_data(x: np.ndarray, sample_size: int, context_length: int, device: torch.device = None, dtype: torch.dtype = None) -> (torch.Tensor, torch.Tensor):
    """
    Args:
        x (np.ndarray): integer array with token IDs to sample from
        sample_size (int): amount of samples to generate
        context_length (int): context length
    Returns:
        (torch.Tensor, torch.Tensor): the sampled input sequences and the corresponding next-token targets, each (sample_size, context_length)
    """

    x = torch.from_numpy(x)
    max_start_exclusive = len(x) - context_length
    
    starts = torch.randint(0, max_start_exclusive, (sample_size,))
    offsets = torch.arange(context_length).unsqueeze(0)  # (1, context_length)
    idx = starts.unsqueeze(1) + offsets  # (sample_size, context_length)

    samples = x[idx]
    targets = x[idx + 1]
    return samples.to(device=device, dtype=dtype), targets.to(device=device, dtype=dtype)
    #test: uv run pytest -k test_get_batch

class CustomDataset(Dataset):
    def __init__(self, tokens_path: str, context_length: int = 256, sample_size: int = 10):
        token_ids = np.load(tokens_path, mmap_mode='r')
        self.inputs, self.targets = sample_data(token_ids, sample_size, context_length)
    
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def create_dataloader(tokens_path: str, context_length: int = 256, shuffle: bool = True, batch_size: int = 4, sample_size: int = 10, drop_last: bool = True):
    dataset = CustomDataset(tokens_path, context_length, sample_size)
    return DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, num_workers=2, shuffle=shuffle)




