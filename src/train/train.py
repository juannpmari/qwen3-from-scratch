# Training loop

import argparse
from src.qwen3.transformer import Transformer
from src.train.optimizer import AdamW
from src.train.loss import compute_cross_entropy
from src.train.checkpointing import save_checkpoint, load_checkpoint
from src.train.optimizer import clip_gradients, cosine_annealing_lr_scheduler


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    transformer = Transformer(args.d_model, args.num_heads, args.max_seq_len, args.device)
    optimizer = AdamW(transformer.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.num_epochs):
        for batch in sample_data(args.batch_size, args.max_seq_len, args.device):
            optimizer.zero_grad()
            transformer(batch)
            loss = compute_cross_entropy(transformer.logits, batch)
            loss.backward()
            clip_gradients(transformer.parameters(), args.max_grad_norm)
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                save_checkpoint(transformer, optimizer, epoch, args.checkpoint_dir)

                val_loss = compute_cross_entropy(transformer.logits, batch)
                print(f"Validation Loss: {val_loss.item()}")
                