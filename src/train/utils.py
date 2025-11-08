import torch.utils.data
from src.preprocessing.dataloader import create_dataloader
import os
import matplotlib.pyplot as plt

def load_data(args, mode="train") -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    if mode == "train":
        train_paths = [os.path.join(args.train_path, f) for f in os.listdir(args.train_path)]
        val_paths = [os.path.join(args.val_path, f) for f in os.listdir(args.val_path)]
        train_dl = create_dataloader(train_paths, batch_size=args.batch_size, context_length=args.context_length, sample_size=args.sample_size)
        val_dl = create_dataloader(val_paths, batch_size=args.batch_size, context_length=args.context_length, sample_size=args.sample_size//10)
        return train_dl, val_dl
    elif mode == "test":
        test_paths = [os.path.join(args.test_path, f) for f in os.listdir(args.test_path)]
        test_dl = create_dataloader(test_paths, batch_size=args.batch_size, context_length=args.context_length, sample_size=args.sample_size)
        return test_dl

def plot_losses(epochs_seen, loss: list[float], save_path:str):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, loss, label="Training loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(save_path)