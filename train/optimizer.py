import torch
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
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.95, eps=1e-08, w_decay = 0.01):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        default = {"lr": lr, "b1": b1, "b2": b2, "eps": eps, "w_decay": w_decay}
        super().__init__(params, default)
        for p in params:
            self.state[p]['m'] = torch.zeros_like(p)
            self.state[p]['v'] = torch.zeros_like(p)

    def step(self, closure:Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            b1 = group['b1']
            b2 = group['b2']
            eps = group['eps']
            w_decay = group['w_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                m = b1*state['m'] + (1-b1)*grad
                v = b2*state['v'] + (1-b2)*grad**2
                state['m'] = m
                state['v'] = v

                t = state.get("t", 1)
                lr = lr*math.sqrt(1-b2**t)/(1-b1**t)
                p.data -= lr*m/math.sqrt(v+eps)
                p.data -= lr*w_decay*p.data
                state["t"] = t + 1
        return loss


def cosine_annealing_lr_scheduler(t, max_lr, min_lr, Tw, Tc):
    """
    t: current iteration
    max_lr: maximum learning rate
    min_lr: minimum learning rate
    Tw: warmup steps
    Tc: cosine annealing steps
    """
    if t < Tw:
        return t/Tw * max_lr
    elif t >= Tw and t <= Tc:
        return min_lr + 0.5*(1+math.cos(math.pi*(t-Tw)/(Tc-Tw)))*(max_lr-min_lr)
    elif t > Tc:
        return min_lr
        
        

class GradientClipper:
    pass