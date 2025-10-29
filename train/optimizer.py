from torch.optim import Optimizer
from typing import Optional, Callable
import math

class SGD(Optimizer):
    """
    Stochastic gradient descent optimizer, implements learning rate decay.
    """
    def __init__(self, params, lr=1e-3):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        default = {"lr": lr}
        super().__init__(params, default)


    def step(self, closure:Optional[Callable] = None):
        loss = None if closure is None else closure() #not used now but required by torch interface.
        for group in self.param_groups:
            lr = group['lr'] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr/math.sqrt(t+1) * grad # Update weight tensor in-place (SGD variant that implements learning rate decay).
                state["t"] = t + 1 # Increment iteration number.
        return loss



class AdamW(Optimizer):
    """
    AdamW optimizer.
    """
    def __init__(self, params, lr=1e-3):
        super().__init__(params, lr=lr)
        pass

    def step(self, closure:Optional[Callable] = None):
        pass

class LRScheduler:
    pass

class GradientClipper:
    pass