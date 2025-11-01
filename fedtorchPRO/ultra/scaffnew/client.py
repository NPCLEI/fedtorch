import torch

from ...core.client import Client
from fedtorchPRO import ModelTool,SuperConfig

class scaffnew_gradhandle:

    def __init__(self,hi,ars,nk,device) -> None:
        ModelTool.to(hi,device)
        self.hi = hi
        batchsize = ars.batch_size
        epoch = ars.epoch
        self.alpha = 1.0/(epoch * (nk//batchsize  + 1) )

    @torch.no_grad()
    def __call__(self,net):
        for g,hi in zip(net.parameters(),self.hi.parameters()):
            g.grad.sub_(hi,alpha = self.alpha)

    def cpu(self):
        ModelTool.cpu(self.hi)

class SCAFFNEWClient(Client):

    def _init(self):
        self.inited = False

    def _init_param(self,args : SuperConfig):
        if self.inited:
            return
        self.hi = args.ZeroNET()
        self.model_ = args.inimodel()
        self.inited = True
    
    @torch.no_grad()
    def updatehi(self,args:SuperConfig,model):
        global_model = args.server.acquire_global_model(self.cid)
        # p = args.scaffnew
        for hi,x,g in zip(self.hi.parameters(),model.parameters(),global_model.parameters()):
            hi.mul_(0.9)
            hi.add_(g - x,alpha=1-0.9)

    def traincore(self,args:SuperConfig,model,device):
        self._init_param(args)
        trainer = args.TRAINER(
            model,
            train_dataset = self.train_dataset,
            test_dataset = None,
            device = device,
            **args.client_trainer_arguments
        )
        trainer.gradhandle = scaffnew_gradhandle(self.hi,args,self.nk,device)
        res = trainer.fit()
        trainer.gradhandle.cpu()
        self.updatehi(args,res[0])
        return res

