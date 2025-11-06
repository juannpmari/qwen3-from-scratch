# Training loop

import argparse
from src.qwen3.transformer import Transformer
from src.train.optimizer import AdamW
from src.train.loss import compute_cross_entropy
from src.train.checkpointing import save_checkpoint, load_checkpoint
from src.train.optimizer import clip_gradients, cosine_annealing_lr_scheduler


def train(model:nn.Module, optimizer:nn.optim.Optimizer, args: argparse.Namespace, train_loader, val_loader):
    tokens_seen, global_steps = 0,0
    for epoch in range(args.num_epochs):
        for input, target in train_loader:
            optimizer.zero_grad()
            logits = transformer(input)
            loss = compute_cross_entropy(logits, target)
            loss.backward() # CHECK
            clip_gradients(transformer.parameters(), args.max_grad_norm)
            optimizer.step()
            tokens_seen += len(input)
            global_steps += 1
            
            if global_steps % 10 == 0:
                print(f"Epoch {epoch}, Global Steps: {global_steps}")
                print(f"Training loss: {loss.item()}")
                val_logits = transformer(val_loader)
                val_loss = compute_cross_entropy(val_logits, val_loader) #CHECK: use torch.no_grad() ?
                print(f"Validation Loss: {val_loss.item()}")
                
                save_checkpoint(transformer, optimizer, epoch, args.checkpoint_dir)
    return loss, val_loss, tokens_seen