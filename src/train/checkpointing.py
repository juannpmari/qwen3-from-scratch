import torch

def save_checkpoint(model, optimizer, iteration, out):
    torch.save({
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, out)

def load_checkpoint(src, model, optimizer):
    data = torch.load(src)
    model.load_state_dict(data['model_state_dict'])
    optimizer.load_state_dict(data['optimizer_state_dict'])
    return data['iteration']

#Test: uv run pytest -k test_checkpointing