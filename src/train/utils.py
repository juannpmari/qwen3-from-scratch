import torch.utils.data
from src.preprocessing.dataloader import create_dataloader
import os

def load_data(args, mode="train") -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    if mode == "train":
        train_paths = [os.path.join(args.train_path, f) for f in os.listdir(args.train_path)]
        val_paths = [os.path.join(args.val_path, f) for f in os.listdir(args.val_path)]
        train_dl = create_dataloader(train_paths, args.batch_size, args.context_length)
        val_dl = create_dataloader(val_paths, args.batch_size, args.context_length)
        return train_dl, val_dl
    elif mode == "test":
        test_paths = [os.path.join(args.test_path, f) for f in os.listdir(args.test_path)]
        test_dl = create_dataloader(test_paths, args.batch_size, args.context_length)
        return test_dl