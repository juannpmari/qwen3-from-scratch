import yaml
import argparse
from argparse import Namespace
from src.preprocessing.tokenizer import BPETokenizer
from src.preprocessing.dataloader import sample_data
from src.train.train import train
from src.inference.generate import generate
from src.qwen3.transformer import Transformer
from src.train.optimizer import AdamW
from torch.utils.data import DataLoader

def load_data(args) -> DataLoader: #TODO: check if this should be moved to /preprocessing, or split in 1.tokenizer and 2.dataloader
    tokenizer = BPETokenizer.from_files(args.vocab_file, args.merges_file, args.special_tokens)
    samples, targets = sample_data(args.dataset_path, args.batch_size, args.max_seq_len, args.device)
    return DataLoader(samples, targets, batch_size=args.batch_size, shuffle=True) # CHECK

def train_model(args):
    print("training model...")
    dataloader = load_data(args)
    model = Transformer(vocab_size=args.vocab_size, num_layers=args.num_layers, max_seq_len=args.max_seq_len, hidden_size=args.d_model, dff = args.dff, gka_ratio=args.gka_ratio, num_heads=args.num_heads,  device=args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    train_loss, val_loss, tokens_seen = train(model, optimizer, args, dataloader)  # TODO: samples and targets should come as a torch Dataloader
    
    print(f"Final train loss: {train_loss}")
    print(f"Final validation loss: {val_loss}")
    print(f"Tokens seen: {tokens_seen}")

def generate(args):
    print("Generating text...")
    model = Transformer(vocab_size=args.vocab_size, num_layers=args.num_layers, max_seq_len=args.max_seq_len, hidden_size=args.d_model, dff = args.dff, gka_ratio=args.gka_ratio, num_heads=args.num_heads,  device=args.device)
    model.load_state_dict(torch.load(args.checkpoint_dir))
    model.eval()
    
    tokenizer = BPETokenizer.from_files(args.vocab_file, args.merges_file, args.special_tokens)
    prompt = "Hello, how are you?"
    max_tokens = 10
    temperature = 0.7
    top_p = 0.9
    
    generated_text = generate(model, tokenizer, prompt, max_tokens, temperature, top_p)
    print("Generated text:", generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3 Training Script")
    parser.add_argument("--config", type=str, default="experiments/base_config.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    
    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Flatten the nested config structure into a single namespace
    args = Namespace(
        mode=config["mode"],
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        gka_ratio=config["model"]["gka_ratio"],
        dff=config["model"]["dff"],
        dataset_path=config["training"]["dataset_path"],
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        num_epochs=config["training"]["num_epochs"],
        max_seq_len=config["training"]["max_seq_len"],
        max_grad_norm=config["training"]["max_grad_norm"],
        checkpoint_dir=config["training"]["checkpoint_dir"],
        device=config["device"]
    )

    if args.mode == "train":
         train_model(args)
 
    elif args.mode == "inference":
        generate(args)
