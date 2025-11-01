import torch
from torch_optimizer import Yogi,Lamb
from fedtorchPRO.ultra.opt.server import FedOPT
from fedtorchPRO.superconfig import SuperConfig

class FADAS(FedOPT):

    def select_opt(self, model, lr , optname = None):   
        return torch.optim.Adam(model.parameters(),lr = lr,amsgrad=True)

    @torch.no_grad()
    def aggregate(self):

        for params in zip(
                self.global_model.parameters(True),
                *[self[cid].model.parameters(True) for cid in self.selected_ids]
            ):
            grad = (params[0].detach() - params[1]) * self.weight[0]
            for idx in range(2,len(params)):
                grad.add_(params[0].detach() - params[idx],alpha = self.weight[idx-1])
            gn,M = grad.norm(2),len(self.selected_history)
            eta = max(1,gn/(2*M*(gn+1e-8)))
            params[0].data.add_(grad,alpha = -eta) 

        return self.global_model
