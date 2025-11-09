import argparse
from src.train.loss import compute_cross_entropy_batch
from src.train.checkpointing import save_checkpoint
from src.train.optimizer import clip_gradients
import torch.nn as nn
import torch.optim as optim
import torch
import os
from datetime import datetime
from torch.utils.data import DataLoader
from typing import Tuple, List


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    train_dl: DataLoader,
    val_dl: DataLoader,
    device: torch.device = None,
) -> Tuple[List[float], List[float], int, int, str]:
    """
    Training loop for the model.
    Args:
        model: model to train
        optimizer: optimizer to use
        args: arguments
        train_dl: training data loader
        val_dl: validation data loader
        device: device to run on
    Returns:
        track_train_loss: list of training losses
        track_val_loss: list of validation losses
        tokens_seen: number of tokens seen
        global_steps: number of global steps
        checkpoint_dir: directory where checkpoints are saved
    """
    tokens_seen, global_steps = 0, 0
    track_train_loss = []
    track_val_loss = []
    checkpoint_dir = os.path.join(
        args.checkpoint_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    for epoch in range(args.num_epochs):
        for input, target in train_dl:
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(input)
            loss = compute_cross_entropy_batch(logits, target)
            loss.backward()  # CHECK; here it gets stuck
            clip_gradients(model.parameters(), args.max_grad_norm)
            optimizer.step()
            tokens_seen += len(input)
            global_steps += 1
            track_train_loss.append(loss.item())
            print(f"Global Steps: {global_steps}, Training loss: {loss.item()}")

            if global_steps % args.checkpoint_interval == 0:
                print(f"Epoch {epoch}, Global Steps: {global_steps}")
                val_loss = 0
                for input_val, target_val in val_dl:
                    input_val = input_val.to(device)
                    target_val = target_val.to(device)
                    val_logits = model(input_val)
                    val_loss += compute_cross_entropy_batch(val_logits, target_val)
                val_loss /= len(val_dl)
                print(f"Validation Loss: {val_loss.item()}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt"),
                )
                track_val_loss.append(val_loss.item())
    return track_train_loss, track_val_loss, tokens_seen, global_steps, checkpoint_dir
