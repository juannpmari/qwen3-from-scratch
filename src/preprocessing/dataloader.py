import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocessing.tokenizer import BPETokenizer
from typing import List
import tiktoken


def sample_data(x: np.ndarray, sample_size: int, context_length: int) -> (torch.Tensor, torch.Tensor):
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
    return samples, targets

class CustomDataset(Dataset):
    def __init__(self, txt_file: List[str], context_length: int = 256, sample_size: int = 10):
        data = []
        for txt in txt_file:
            with open(txt, 'r') as f:
                data.append(f.read())
        self.context_length = context_length
        # tokenizer = BPETokenizer()
        # token_ids = tokenizer.encode_iterable(data)
        tokenizer = tiktoken.get_encoding("gpt2")
        token_ids = []
        for txt in data: #TODO: optimize this for large datasets
            token_ids.extend(tokenizer.encode_ordinary(txt))#, allowed_special={"<|endoftext|>"}))
        token_ids = np.array(token_ids) #Check

        self.inputs, self.targets = sample_data(token_ids, sample_size, context_length)
    
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def create_dataloader(txt_file: List[str], context_length: int = 256, shuffle: bool = True, batch_size: int = 4, sample_size: int = 10, drop_last: bool = True):
    dataset = CustomDataset(txt_file, context_length, sample_size)
    return DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, num_workers=2, shuffle=shuffle)


#TODO: Fix and replace by yield to return a generator
# Investigate how to do this more efficiently
# Use np.memmap or the flag mmap_mode='r' to np.load when loading the dataset


#test: uv run pytest -k test_get_batch