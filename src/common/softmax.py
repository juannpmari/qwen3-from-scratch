import torch

def softmax(x: torch.tensor, i:int) -> torch.tensor:
    maximum, _ = torch.max(x, dim=i, keepdim=True)
    x = x - maximum
    softmax = torch.exp(x) / torch.sum(torch.exp(x), dim=i, keepdim=True)
    return softmax

# NOTE: this passes the test but in reality doesn't support very large inputs, it overflows
    