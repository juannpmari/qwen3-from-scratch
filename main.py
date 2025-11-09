import yaml
import argparse
from argparse import Namespace
from src.preprocessing.tokenizer import BPETokenizer
from src.train.utils import load_data, plot_losses
from src.train.train import train
from src.inference.generate import generate_text
from src.qwen3.transformer import Transformer
from src.train.optimizer import AdamW
from src.train.checkpointing import load_checkpoint
import tiktoken
import time
import torch

DTYPE_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def train_model(args):
    print("training model...")
    train_dl, val_dl = load_data(args, mode="train")
    model = Transformer(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        context_length=args.context_length,
        hidden_dim=args.d_model,
        dff=args.dff,
        gka_ratio=args.gka_ratio,
        num_heads=args.num_heads,
    )
    device = args.device
    dtype = DTYPE_MAP.get(args.dtype, torch.float32)
    model.to(device=device, dtype=dtype)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    timer = time.time()
    train_loss, val_loss, tokens_seen, global_steps, checkpoint_dir = train(
        model, optimizer, args, train_dl, val_dl, device
    )
    timer = time.time() - timer

    with open(f"{checkpoint_dir}/config.yaml", "w") as f:
        yaml.dump(args, f)

    with open(f"{checkpoint_dir}/summary.txt", "w") as f:
        f.write(f"Final train loss: {train_loss[-1]}\n")
        f.write(f"Final validation loss: {val_loss[-1]}\n")
        f.write(f"Tokens seen: {tokens_seen}\n")
        f.write(f"Time taken: {timer}\n")
        f.write(f"Global steps: {global_steps}\n")

    print(f"Final train loss: {train_loss}")
    print(f"Final validation loss: {val_loss}")
    print(f"Tokens seen: {tokens_seen}")
    print(f"Time taken: {timer}")
    plot_losses(range(global_steps), train_loss, f"{checkpoint_dir}/train_loss.png")
    # plot_losses(range(global_steps), val_loss, f"{checkpoint_dir}/val_loss.png")


def generate(args):
    print("Generating text...")
    # test_dl = load_data(args, mode="test")
    model = Transformer(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        context_length=args.context_length,
        hidden_dim=args.d_model,
        dff=args.dff,
        gka_ratio=args.gka_ratio,
        num_heads=args.num_heads,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    src = args.weights_dir
    iteration = load_checkpoint(src, model, optimizer)
    device = args.device
    dtype = DTYPE_MAP.get(args.dtype, torch.float32)
    model.to(device=device, dtype=dtype)
    model.eval()

    prompt = """I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that, in the height of his glory, he had dropped his painting, married a rich widow, and established himself in a villa on the Riviera. (Though I rather thought it would have been Rome or Florence.)

    "The height of his glory"--"""
    # tokenizer = BPETokenizer.from_files(args.vocab_file, args.merges_file, args.special_tokens)
    tokenizer = tiktoken.get_encoding("gpt2")
    max_tokens = args.max_tokens
    temperature = args.temperature
    top_p = args.top_p

    generated_text = generate_text(
        model, tokenizer, prompt, max_tokens, temperature, top_p, device
    )
    print("Generated text:", generated_text)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Qwen3 Training Script")
    # parser.add_argument("--config", type=str, default="experiments/base_config.yaml", help="Path to the configuration file")
    # args = parser.parse_args()

    # Load configuration from YAML file
    with open("experiments/base_train.yaml", "r") as f:
        # with open("experiments/base_inf.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Flatten the nested config structure into a single namespace
    if config["mode"] == "train":
        args = Namespace(
            mode=config["mode"],
            vocab_size=config["model"]["vocab_size"],
            d_model=config["model"]["d_model"],
            num_heads=config["model"]["num_heads"],
            num_layers=config["model"]["num_layers"],
            gka_ratio=config["model"]["gka_ratio"],
            dff=config["model"]["dff"],
            train_path=config["training"]["train_path"],
            val_path=config["training"]["val_path"],
            batch_size=config["training"]["batch_size"],
            learning_rate=config["training"]["learning_rate"],
            num_epochs=config["training"]["num_epochs"],
            context_length=config["training"]["context_length"],
            max_grad_norm=config["training"]["max_grad_norm"],
            checkpoint_dir=config["training"]["checkpoint_dir"],
            checkpoint_interval=config["training"]["checkpoint_interval"],
            sample_size=config["training"]["sample_size"],
            device=config["device"],
            dtype=config["training"]["dtype"],
        )
    elif config["mode"] == "inference":
        args = Namespace(
            mode=config["mode"],
            vocab_size=config["model"]["vocab_size"],
            d_model=config["model"]["d_model"],
            num_heads=config["model"]["num_heads"],
            num_layers=config["model"]["num_layers"],
            gka_ratio=config["model"]["gka_ratio"],
            dff=config["model"]["dff"],
            weights_dir=config["inference"]["weights_dir"],
            context_length=config["inference"]["context_length"],
            max_tokens=config["inference"]["max_tokens"],
            temperature=config["inference"]["temperature"],
            top_p=config["inference"]["top_p"],
            learning_rate=config["inference"]["learning_rate"],
            sample_size=config["inference"]["sample_size"],
            device=config["device"],
            dtype=config["dtype"],
        )

    if args.mode == "train":
        train_model(args)

    elif args.mode == "inference":
        generate(args)
