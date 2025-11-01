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

import math
from typing import Optional, Callable

import torch
from torch.optim import Optimizer

# Works but created by chatgpt
# class AdamW(Optimizer):
#     """
#     A lightweight, correct implementation of the AdamW optimizer (decoupled weight decay).

#     Usage:
#         optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

#     This implementation follows the standard bias-correction and decoupled weight decay
#     behavior found in PyTorch's official AdamW.
#     """

#     def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
#         if lr <= 0.0:
#             raise ValueError(f"Invalid learning rate: {lr}")
#         if not 0.0 <= eps:
#             raise ValueError(f"Invalid eps value: {eps}")

#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
#         super().__init__(params, defaults)

#     @torch.no_grad()
#     def step(self, closure: Optional[Callable] = None):
#         """
#         Performs a single optimization step.

#         Args:
#             closure (callable, optional): A closure that reevaluates the model and returns the loss.
#         Returns:
#             loss (optional): The value returned by the closure, if provided.
#         """
#         loss = None
#         if closure is not None:
#             # Re-evaluate the model under grad enabled context
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             lr = group["lr"]
#             b1, b2 = group["betas"]
#             eps = group["eps"]
#             wd = group["weight_decay"]

#             for p in group["params"]:
#                 if p.grad is None:
#                     continue

#                 grad = p.grad
#                 if grad.is_sparse:
#                     # Sparse gradients are not supported by this AdamW implementation.
#                     raise RuntimeError("AdamW does not support sparse gradients")

#                 # Ensure state initialization (lazy)
#                 state = self.state.setdefault(p, {})
#                 if len(state) == 0:
#                     state["step"] = 0
#                     state["m"] = torch.zeros_like(p.data)
#                     state["v"] = torch.zeros_like(p.data)

#                 m = state["m"]
#                 v = state["v"]

#                 state["step"] += 1
#                 t = state["step"]

#                 # Update biased first and second moment estimates
#                 m.mul_(b1).add_(grad, alpha=1 - b1)
#                 v.mul_(b2).addcmul_(grad, grad, value=1 - b2)

#                 # Compute bias-corrected step size
#                 bias_correction1 = 1 - b1 ** t
#                 bias_correction2 = 1 - b2 ** t
#                 # small epsilon added to denominator to avoid potential divide-by-zero
#                 step_size = lr * math.sqrt(bias_correction2) / (bias_correction1 + 1e-16)

#                 # Decoupled weight decay (AdamW): scale parameters by (1 - lr * weight_decay)
#                 if wd != 0:
#                     p.data.mul_(1 - lr * wd)

#                 # Parameter update: p = p - step_size * m / (sqrt(v) + eps)
#                 denom = v.sqrt().add_(eps)
#                 p.data.addcdiv_(m, denom, value=-step_size)

#         return loss

class AdamW(Optimizer):
    """
    AdamW optimizer.
    """
    def __init__(self, params, lr=1e-3, betas =(0.9, 0.95), eps=1e-08, weight_decay = 0.01):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        b1, b2 = betas
        default = {"lr": lr, "b1": b1, "b2": b2, "eps": eps, "weight_decay": weight_decay}
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
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                m = b1*self.state[p]['m'] + (1-b1)*grad
                v = b2*self.state[p]['v'] + (1-b2)*grad**2
                self.state[p]['m'] = m
                self.state[p]['v'] = v

                t = state.get("t", 1)
                lr = lr*math.sqrt(1-b2**t)/(1-b1**t)
                p.data -= lr*m/math.sqrt(v+eps)
                p.data -= lr*weight_decay*p.data
                self.state[p]["t"] = t + 1
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
        
        

def clip_gradients(parameters, max_norm, eps = 1e-6):
    """
    parameters: list of parameters
    max_norm: maximum norm of gradients
    eps: small number to avoid division by zero
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
    clip_coef = max_norm / (total_norm + eps)
    for p in parameters:
        if p.grad is not None:
            if p.grad.norm() > max_norm:
                p.grad.data = p.grad.data * clip_coef