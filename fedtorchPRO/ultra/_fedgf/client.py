import torch
from ...core import client

class fedgfhandle:
    def __init__(self,Cs,rho):
        self.cs = Cs
        self.rho = rho

    @torch.no_grad()
    def __call__(self,net,wr):
        for c,pm in zip(self.cs,net.parameters()):
            grad = pm.grad
            pm.add_(grad/(grad.norm(2)+1e-8),alpha = self.rho)

class FedGFClient(client):
    pass