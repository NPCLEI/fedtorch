import torch

from threading import Lock
from fedtorchPRO.core.client import Client
from fedtorchPRO import SuperConfig,ModelTool
from fedtorchPRO.core.cache import ModelManager

# prox: f + 1/2 (x - x0)^2
# df + x - x0 = 0

class proxgrad:

    @torch.no_grad()
    def __init__(self,args:SuperConfig,ornet,device) -> None:
        self.ornet = ornet
        self.mu = args.prox
        args.TRAINER.to(self.ornet,device)

    @torch.no_grad()
    def __call__(self,net):
        for pm,opm in zip(net.parameters(),self.ornet.parameters()):
            pm.grad.add_(pm - opm,alpha = self.mu)

ProxLock = Lock()

class ProxClient(Client):

    cache = None
    @staticmethod
    def acquire(args:SuperConfig,cid,source):
        with ProxLock:
            if ProxClient.cache == None:
                ProxClient.cache = ModelManager(args.NET)
        return ProxClient.cache.acquire(args.server.selected_ids.index(cid),source)
       
    def traincore(self,args,model,device):
        handler = proxgrad(args,ProxClient.acquire(args,self.cid,model),device)
        trainer = args.TRAINER(
            model,
            train_dataset = self.train_dataset,
            test_dataset = None,
            device = device,
            **args.client_trainer_arguments
        )
        trainer.gradhandle = handler
        res = trainer.fit()
        ModelTool.cpu(handler.ornet)
        return res
