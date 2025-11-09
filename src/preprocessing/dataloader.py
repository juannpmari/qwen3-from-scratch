import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocessing.tokenizer import BPETokenizer
from typing import List
import tiktoken


def sample_data(
    x: np.ndarray,
    sample_size: int,
    context_length: int,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> (torch.Tensor, torch.Tensor):
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
    return samples.to(device=device, dtype=dtype), targets.to(
        device=device, dtype=dtype
    )
    # test: uv run pytest -k test_get_batch


class CustomDataset(
    Dataset
):  # TODO: this could be an IterableDataset, that streams the data instead of loading it all at once
    """
    Custom dataset for loading tokenized data from a file.
    """

    def __init__(
        self, tokens_path: str, context_length: int = 256, sample_size: int = 10
    ):
        """
        Args:
            tokens_path (str): path to the file containing the tokenized data
            context_length (int, optional): context length. Defaults to 256.
            sample_size (int, optional): amount of samples to generate. Defaults to 10.
        """
        token_ids = np.load(tokens_path, mmap_mode="r")
        self.inputs, self.targets = sample_data(token_ids, sample_size, context_length)

    def __len__(self):
        """
        Returns:
            int: number of samples
        """
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            idx (int): index of the sample
        Returns:
            tuple: (input, target) where input and target are both tensors of shape (context_length)
        """
        return self.inputs[idx], self.targets[idx]


def create_dataloader(
    tokens_path: str,
    context_length: int = 256,
    shuffle: bool = True,
    batch_size: int = 4,
    sample_size: int = 10,
    drop_last: bool = True,
) -> DataLoader:
    """
    Creates a dataloader for the tokenized data.
    Args:
        tokens_path (str): path to the file containing the tokenized data
        context_length (int, optional): context length. Defaults to 256.
        shuffle (bool, optional): whether to shuffle the data. Defaults to True.
        batch_size (int, optional): batch size. Defaults to 4.
        sample_size (int, optional): amount of samples to generate. Defaults to 10.
        drop_last (bool, optional): whether to drop the last batch if it is not full. Defaults to True.
    Returns:
        DataLoader: dataloader for the tokenized data
    """
    dataset = CustomDataset(tokens_path, context_length, sample_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=2,
        shuffle=shuffle,
    )
