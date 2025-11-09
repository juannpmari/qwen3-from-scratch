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

    def step(self, closure: Optional[Callable] = None):
        loss = (
            None if closure is None else closure()
        )  # not used now but required by torch interface.
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= (
                    lr / math.sqrt(t + 1) * grad
                )  # Update weight tensor in-place (SGD variant that implements learning rate decay).
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(Optimizer):
    """
    AdamW optimizer.
    """

    def __init__(
        self,
        params: torch.Tensor,
        lr: float = 1e-3,
        betas=(0.9, 0.95),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        b1, b2 = betas
        default = {
            "lr": lr,
            "b1": b1,
            "b2": b2,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, default)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                if "v" not in state:
                    state["v"] = torch.zeros_like(p)

    @torch.no_grad()  # To avoid tracking by autograd
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.
        Args:
            closure: callable that returns the loss to be minimized
        Returns:
            loss: loss value
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["b1"]
            b2 = group["b2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                self.state[p]["m"] = (
                    b1 * self.state[p]["m"] + (1 - b1) * grad
                )  # same shape as p.data
                self.state[p]["v"] = (
                    b2 * self.state[p]["v"] + (1 - b2) * grad**2
                )  # same shape as p.data
                m = self.state[p]["m"]
                v = self.state[p]["v"]
                t = state.get("t", 1)
                lr_param = lr * math.sqrt(1 - b2**t) / (1 - b1**t)
                p.data -= lr_param * m / (torch.sqrt(v) + eps)
                self.state[p]["t"] = t + 1
                p.data -= lr * weight_decay * p.data

        return loss


# uv run pytest -k test_adamw


def cosine_annealing_lr_scheduler(
    t: int, max_lr: float, min_lr: float, Tw: int, Tc: int
) -> float:
    """
    Cosine annealing learning rate scheduler.
    Args:
        t: current iteration
        max_lr: maximum learning rate
        min_lr: minimum learning rate
        Tw: warmup steps
        Tc: cosine annealing steps
    """
    if t < Tw:
        return t / Tw * max_lr
    elif t >= Tw and t <= Tc:
        return min_lr + 0.5 * (1 + math.cos(math.pi * (t - Tw) / (Tc - Tw))) * (
            max_lr - min_lr
        )
    elif t > Tc:
        return min_lr


def clip_gradients(
    parameters: torch.Tensor, max_norm: float, eps: float = 1e-6
) -> None:
    """
    Clips the gradients of the parameters to a maximum norm.

    Args:
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
