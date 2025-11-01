
import torch

class deltaSGD:

    @torch.no_grad()
    def __init__(self,parameters,lr,gamma):
        self.parameters = [pm for pm in parameters]
        self.lr = lr
        self.gamma = gamma
        self.etas = [torch.tensor(lr) for _ in self.parameters]
        self.thetas = [torch.tensor(lr) for _ in self.parameters]
        self.last_grads = [torch.zeros_like(pm,requires_grad=False) for pm in self.parameters]
    
    @torch.no_grad()
    def zero_grad(self):
        for pm in self.parameters:
            pm.grad = None

    @torch.no_grad()
    def step(self):
        for pm,lg,eta,theta in zip(self.parameters,self.last_grads,self.etas,self.thetas):
            delta = - eta * pm.grad
            pm.add_(delta)
            etak = min(self.gamma * delta.norm(2) / (2 * (pm.grad - lg).norm(2)),eta * (1 + theta)**0.5)
            theta.data = etak / eta
            eta.data = etak
         