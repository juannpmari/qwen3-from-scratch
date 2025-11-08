import torch.utils.data
from src.preprocessing.dataloader import create_dataloader
import os
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from typing import List

def pretokenize(txt_file: List[str], token_path: str = "data/tokens"):
    """
    Tokenize and store training tokens for efficiency
    """
    data = []
    for txt in txt_file:
        with open(txt, 'r') as f:
            data.append(f.read())
    # tokenizer = BPETokenizer()
    # token_ids = tokenizer.encode_iterable(data)
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = []
    for txt in data: #TODO: optimize this for large datasets
        token_ids.extend(tokenizer.encode_ordinary(txt))#, allowed_special={"<|endoftext|>"}))
    token_ids = np.array(token_ids) #Check
    np.save(f"{token_path}/tokens.npy", token_ids)


def load_data(args, mode="train") -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    if mode == "train":
        train_dl = create_dataloader(args.train_path, batch_size=args.batch_size, context_length=args.context_length, sample_size=args.sample_size)
        val_dl = create_dataloader(args.val_path, batch_size=args.batch_size, context_length=args.context_length, sample_size=args.sample_size//10)
        return train_dl, val_dl
    elif mode == "test":
        test_dl = create_dataloader(args.test_path, batch_size=args.batch_size, context_length=args.context_length, sample_size=args.sample_size)
        return test_dl

def plot_losses(epochs_seen, loss: list[float], save_path:str):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, loss, label="Training loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(save_path)