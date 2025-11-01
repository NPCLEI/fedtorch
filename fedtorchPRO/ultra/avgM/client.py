import torch

from fedtorchPRO.core.client import Client
from fedtorchPRO import ModelTool

class momentum:

    @torch.no_grad()
    def __init__(self,beta,grad,device) -> None:
        self.grad = grad
        self.beta = beta
        ModelTool.to(grad,device)

    @torch.no_grad()
    def __call__(self,net:torch.nn.Module):
        for pm,gradi in zip(net.parameters(),self.grad.parameters()):
            pm.grad.mul_(self.beta)
            pm.grad.add_(gradi,alpha = 1 - self.beta)


class avgMClient(Client):
       
    def traincore(self,args,model,device):
        handler = momentum(args.get('fedavgm',0.1),args.server.acquiregrad(self.cid),device)
        trainer = args.TRAINER(
            model,
            train_dataset = self.train_dataset,
            test_dataset = None,
            device = device,
            **args.client_trainer_arguments
        )
        trainer.gradhandle = handler
        res = trainer.fit()
        ModelTool.cpu(handler.grad)
        return res
