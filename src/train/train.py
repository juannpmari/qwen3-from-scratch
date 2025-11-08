import argparse
from src.train.loss import compute_cross_entropy_batch
from src.train.checkpointing import save_checkpoint
from src.train.optimizer import clip_gradients
import torch.nn as nn
import torch.optim as optim
import torch
import os

def train(model:nn.Module, optimizer:optim.Optimizer, args: argparse.Namespace, train_dl, val_dl, device:torch.device = None):
    tokens_seen, global_steps = 0,0
    val_loss = None
    for epoch in range(args.num_epochs):
        for input, target in train_dl:
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(input)
            loss = compute_cross_entropy_batch(logits, target)
            loss.backward() # CHECK
            clip_gradients(model.parameters(), args.max_grad_norm)
            optimizer.step()
            tokens_seen += len(input)
            global_steps += 1
            
            if global_steps % args.checkpoint_interval == 0:
                print(f"Epoch {epoch}, Global Steps: {global_steps}")
                print(f"Training loss: {loss.item()}")
                for input_val, target_val in val_dl:
                    input_val = input_val.to(device)
                    target_val = target_val.to(device)
                    val_logits = model(input_val)
                    val_loss = compute_cross_entropy_batch(val_logits, target_val)
                print(f"Validation Loss: {val_loss.item()}")
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                save_checkpoint(model, optimizer, epoch, os.path.join(args.checkpoint_dir, f"checkpoint_{epoch}.pt"))
            print(f"Training loss: {loss.item()}")
    return loss, val_loss, tokens_seen