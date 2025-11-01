import torch

from fedtorchPRO import FedAVGPlus
from .client import SCAFFOLDClient
from fedtorchPRO.core.cache import ModelManager

class SCAFFOLD(FedAVGPlus):

    @torch.no_grad()
    def _init(self):
        self.CLIENT = SCAFFOLDClient
        self._convert_FedAVG()
        self.server_controls = self.arguments.ZeroNET()
        self.server_controls_mananger = ModelManager(self.arguments.NET)
        self.global_mananger = ModelManager(self.arguments.NET)
        return super()._init()

    def aquiresc(self,cid):
        tid = self.selected_ids.index(cid)
        return self.server_controls_mananger.acquire(tid,self.server_controls)

    def aquiregm(self,cid):
        tid = self.selected_ids.index(cid)
        return self.global_mananger.acquire(tid,self.global_model)

    @torch.no_grad()
    def aggregate(self):

        weights = self.weight

        self.opt.zero_grad()
        for params in zip(self.global_model.parameters(),*[self[cid].model.parameters() for cid in self.selected_ids]):
            grad = 0
            for i,cm in enumerate(params[1:]):
                grad += weights[i] * (params[0] - cm)
            params[0].grad = grad
        self.opt.step()
        
        for params in zip(self.server_controls.parameters(),*[self[cid].ci_delta.parameters() for cid in self.selected_ids]):
            delta = torch.zeros_like(params[0])
            for i,cm in enumerate(params[1:]):
                delta.add_(cm ,alpha = weights[i])
            params[0].add_(delta,alpha = len(self.selected_ids)/len(self))

        return self.global_model
