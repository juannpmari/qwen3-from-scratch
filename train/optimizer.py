from torch.optim import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, lr=lr)
        pass

    def step(self):
        pass

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, lr=lr)
        pass

    def step(self):
        pass

class LRScheduler:
    pass

class GradientClipper:
    pass