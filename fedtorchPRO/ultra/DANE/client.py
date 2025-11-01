import torch
from fedtorchPRO.core.client import Client

# prox: x <- min f + 1/2 (x - x0)^2 + 1/2 (x - x_t-1)^2
# 0 = df + x - x0 + x - x  

class dualproxgrad:

    @torch.no_grad()
    def __init__(self,args,ornet,device) -> None:
        self.xkhat = args.NET()
        self.args = args
        for pm,o in zip(self.xkhat.parameters(),ornet.parameters()):
            pm.data[:] = o
        self.mu = args.prox
        self.lmbda = args.dualprox
        args.TRAINER.to(self.xkhat,device)
        self.counter = 0
        self.nets_stack = []
        self.cache = None

    def _pop(self):
        self.cache = self.nets_stack.pop(0) 
        return self.cache
    
    def _stack(self,net):
        if self.cache:
            cache = self.cache
        else:
            self.nets_stack.append(self.args.NET())
            cache = self.nets_stack[-1]
        for cm,pm in zip(cache.parameters(),net.parameters()):
            cm.data[:] = pm
        self.nets_stack.append(cache)

    def before(self,net):
        self._stack(net)

    def updatexkhat(self):
        pass
       

    @torch.no_grad()
    def __call__(self,net):
        xik = self._pop() if self.counter > 0 else [0 for _ in net.parameters()]
        for pm,xkhat,xiki in zip(net.parameters(),self.xkhat.parameters(),xik):
            pm.grad.add_(pm - xkhat,self.mu)
            pm.grad.add_(pm - xiki ,self.lmbda)
         
class DualProxClient(Client):
    def traincore(self,arguments,model,device):
        handler = dualproxgrad(arguments,model,device)
        trainer = arguments.TRAINER(
            model,
            train_dataset = self.train_dataset,
            test_dataset = None,
            device = device,
            **arguments.client_trainer_arguments
        )
        trainer.gradhandle = handler
        trainer.ontrainhandle = handler.before
        res = trainer.fit()
        arguments.TRAINER.to(handler.ornet,'cpu')
        return res
