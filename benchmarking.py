import argparse
import torch
import timeit
from src.qwen3.transformer import Transformer
from src.train.loss import compute_cross_entropy_batch
import matplotlib.pyplot as plt

def benchmark_llm(args): #CHECK this function
    print("Benchmarking...")
    model = Transformer(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        context_length=args.context_length,
        hidden_dim=args.d_model,
        dff=args.dff,
        gka_ratio=args.gka_ratio,
        num_heads=args.num_heads,
    )
    model.to(args.device)
    model.eval()

    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
    target = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)

    for _ in range(args.warmup_steps): # Until reach a steady state
        model(input_ids)

    def sync_function():
        if args.device == "mps":
            torch.mps.synchronize()
        elif args.device == "cuda":
            torch.cuda.synchronize()

    if args.forward_only:
        sync_function()
        timer = timeit.default_timer()
        with torch.no_grad():
            for _ in range(args.num_steps):
                model(input_ids)

    else:
        sync_function()
        timer = timeit.default_timer()
        for _ in range(args.num_steps):
            logits = model(input_ids)
            loss = compute_cross_entropy_batch(logits, target)
            loss.backward()

    sync_function()
    time_taken = (timeit.default_timer() - timer)/args.num_steps
    print(f"Mean step time taken for {args.num_steps} steps: {time_taken}")
    return time_taken
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3 Benchmarking Script")
    parser.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--context-length", type=int, default=256, help="Context length")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension")
    parser.add_argument("--dff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--gka-ratio", type=int, default=1, help="GQA ratio")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--device", type=str, default="mps", help="Device to run on")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup-steps", type=int, default=25, help="Number of warmup steps")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of steps")
    parser.add_argument("--forward-only", type=bool, default=True, help="Forward only")
    args = parser.parse_args()

    time_taken = []
    for w in range(10):
        args.warmup_steps = w
        time_taken.append(benchmark_llm(args))

    print(time_taken)

    def plot_list(values, filename="plot.png"):
        plt.plot(range(len(values)), values)
        plt.savefig(filename)
        plt.close()

    plot_list(time_taken)
