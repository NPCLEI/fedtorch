import torch

from ..opt.servers import FedAVGPlus
from fedtorchPRO.core.cache import ModelManager
from .client import avgMClient
# MOMENTUM BENEFITS NON-IID FEDERATED LEARNING SIMPLY AND PROVABLY

class FedAVGM(FedAVGPlus):

    def _init(self):
        super()._init()
        self.grad = self.arguments.ZeroNET()
        self.gradmanager = ModelManager(self.arguments.NET)
        self.CLIENT = avgMClient

    def acquiregrad(self,cid):
        tid = self.selected_ids.index(cid)
        return self.gradmanager.acquire(tid,self.grad)

    @torch.no_grad()
    def aggregate(self):
        glbmd = super().aggregate()
        for pm,gm in zip(glbmd.parameters(),self.grad.parameters()):
            gm.copy_(pm.grad)
        return glbmd
