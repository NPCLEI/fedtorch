import torch
from fedtorchPRO.core.client import Client

# prox: x <- min f + 1/2 (x - x0)^2 + 1/2 (x - x_t-1)^2
# 0 = df + x - x0 + x - x  

class dualproxgrad:

    @torch.no_grad()
    def __init__(self,args,ornet) -> None:
        self.xkhat = ornet
        self.args = args
        self.mu = args.prox
        self.lmbda = args.dualprox
        self.counter = 0
        self.nets_stack = []
        self.cache = None
        
    @torch.no_grad()
    def _pop(self):
        self.cache = self.nets_stack.pop(0)
        return self.cache
    
    @torch.no_grad()
    def _stack(self,net):
        if self.cache:
            cache = self.cache
            self.nets_stack.append(cache)
        else:
            self.nets_stack.append(self.args.NET())
            cache = self.nets_stack[-1]
        self.args.TRAINER.to(self.cache,net)
        for cm,pm in zip(cache.parameters(),net.parameters()):
            cm.data[:] = pm

    @torch.no_grad()
    def __call__(self,net):
        self.args.TRAINER.to(self.xkhat,net)
        if self.counter > 0:
            xik = self._pop()
            self.args.TRAINER.to(xik,net)
            for pm,xkhat,xiki in zip(net.parameters(),self.xkhat.parameters(),xik.parameters()):
                pm.grad.add_(pm - xkhat,alpha = self.mu)
                pm.grad.add_(pm - xiki ,alpha = self.lmbda)
        else:
            for pm,xkhat in zip(net.parameters(),self.xkhat.parameters()):
                pm.grad.add_(pm - xkhat,alpha = self.mu)
        self.counter += 1

    def cpu(self):
        for net in self.nets_stack:
            self.args.TRAINER.to(net,'cpu')
        self.args.TRAINER.to(self.cache,'cpu')
        

class DualProxClient(Client):

    def _init(self):
        self.selfmodel = None
        self.gradhandle = None

    def traincore(self,arguments,model,device):
        if self.selfmodel == None:
            self.selfmodel = arguments.NET()
            self.handler = dualproxgrad(arguments,model)
        trainer = arguments.TRAINER(
            self.selfmodel,
            train_dataset = self.train_dataset,
            test_dataset = None,
            device = device,
            **arguments.client_trainer_arguments
        )
        trainer.ontrainhandle = self.handler._stack
        trainer.gradhandle = self.handler
        res = trainer.fit()
        arguments.TRAINER.to(self.selfmodel,'cpu')
        self.handler.cpu()
        return res
