import torch

from fedtorchPRO import SuperConfig,ModelTool,ModelTool
from fedtorchPRO.core.client import Client

# prox: x <- min f + 1/2 (x - x0)^2 + 1/2 (x - x_t-1)^2
# 0 = df + x - x0 + x - x        

class DualHandle:

    @torch.no_grad()
    def __init__(self,args:SuperConfig,x0,device):
        self.xt_1 = args.NET().to(device)
        for pm in self.xt_1.parameters():
            pm.fill_(0.0)
            pm.requires_grad_(False)
        self.xt = args.NET().to(device)
        self.x0 = args.NET().to(device)
        args.copynet(x0,self.x0)
        self.count = 0
        self.mu = args.get('prox',0.01)
        self.lmbda = args.get('dualprox',0.1)
        ModelTool.close_grad(self.x0)
        ModelTool.close_grad(self.xt)
        ModelTool.close_grad(self.xt_1)

    def cpu(self):
        ModelTool.cpu(self.xt_1)
        ModelTool.cpu(self.xt)
        ModelTool.cpu(self.x0)
    
    @torch.no_grad()
    def copy(self,target,source):
        for tar,om in zip(target.parameters(),source.parameters()):
            tar.copy_(om)
        return target

    def record(self,xt):
        self.copy(self.xt_1,xt)
        self.copy(self.xt,xt)

    def __call__(self, net):
        for pm,xkhat,xiki in zip(net.parameters(),self.x0.parameters(),self.xt_1.parameters()):
            pm.grad.add_(pm - xkhat,alpha = 0.5 * self.mu)
            pm.grad.add_(pm - xiki ,alpha = 0.5 * self.lmbda)

class DualProxMClient(Client):

    def traincore(self,arguments:SuperConfig,model,device):
        handler = DualHandle(arguments,model,device)
        trainer = arguments.TRAINER(
            model,
            train_dataset = self.train_dataset,
            test_dataset = None,
            device = device,
            clip_grad = 1.0,
            **arguments.client_trainer_arguments
        )
        trainer.ontrainhandle = handler.record
        trainer.gradhandle = handler
        res = trainer.fit()
        # arguments.TRAINER.to(self.selfmodel,'cpu')
        handler.cpu()
        return res
