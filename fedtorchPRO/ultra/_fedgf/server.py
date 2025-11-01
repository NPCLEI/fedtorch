import torch

from ..opt.server import FedOPT
from .client import FedGFClient
from threading import Lock
from fedtorchPRO.core.cache import ModelManager

class FedGF(FedOPT):
    
    def _init(self):
        self.CLIENT = FedGFClient
        self._convert_FedAVG()
        self.paramC = []
        self.cscflag = -2
        self.cslock = Lock()
        self.arguments.wr_manage = ModelManager(self.arguments.NET)
        self.wr = self.arguments.NET()
        return super()._init()
    
    @torch.no_grad()
    def aggregate(self):
        """
            fedavg
        """

        weight = self.weight

        self.opt.zero_grad()
        tempC = []
        for params in zip(
                self.global_model.parameters(True),self.wr.parameters(True),
                *[self[cid].model.parameters(True) for cid in self.selected_ids]
            ):
            grad = (params[0].detach() - params[2]) * weight[0]
            D = grad.norm(p=2)
            for idx in range(3,len(params)):
                delta = params[0].detach() - params[idx]
                alpha = weight[idx-2]
                D.add_(delta,alpha = alpha)
                grad.add_(delta,alpha = alpha)
            params[0].grad = grad
            params[1].copy_(params[0])
            params[1].add_(grad/(grad.norm(p=2)+1e-8),alpha = self.arguments.fedgf)
            tempC.append((D>self.arguments.fedgf).int().item())
        self.opt.step()
        self.paramC.append(tempC)
        
        return self.global_model
    
    @property
    def Cs(self):
        self.cslock.acquire()
        if self.cscflag != self.cur_comic:
            self._Cs = torch.tensor(self.paramC,dtype=torch.float32).sum(0).tolist()
        self.cslock.release()    
        return self._Cs