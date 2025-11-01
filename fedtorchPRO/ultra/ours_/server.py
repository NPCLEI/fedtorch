

# grads = [gradi]
# update grads : i in seleted_ids
# delta <- beta^(t-t') gradi 
# m_t <- beta * grad + (1 - beta) * m_t-1
# v_t <- beta1 1/|grads| * ||grads|| + beta2 v_t-1
# ams

import torch
# from fedtorchPRO.core.server import Server
# from fedtorchPRO.core.utils import ModelTool
from ..opt.servers import FedAdagrad,FedADAM,FedOPT,FedSGD

class Ours(FedADAM):

    def _init(self):
        self.arguments.config['fedopt']['betas'] = (0.,0.999)
        super()._init()

    def _onrun(self):
        super()._onrun()
        self.grads = {}
        self.grdst = [0 for _ in range(len(self))]
        self.grad = self.arguments.ZeroNET()
        # self.mt = self.arguments.ZeroNET()
        self.beta1 = self.arguments.get('ours_beta',0.9)
        self.lmbda = self.arguments.get('ours_grad_decay',0.9)

    @torch.no_grad()
    def update_grads(self):
        # Perfect !

        for id,cid in enumerate(self.selected_ids):
            for x,xc,gc in zip(self.global_model.parameters(True),self[cid].model.parameters(True),self.grad.parameters(True)):
                gc.mul_(self.lmbda)
                gc.add_(x - xc)
        
    # @torch.no_grad()
    # def update_grads(self):
    #     # Good ! but memory need.
    #     keys = [cid for cid in self.grads.keys()]
    #     for grads in zip(*[self.grads[cid].parameters(True) for cid in keys]):
    #         for grad in grads:
    #             grad.mul_(self.lmbda)

    #     for id,cid in enumerate(self.selected_ids):
    #         if cid not in self.grads:
    #             self.grads[cid] = self.arguments.ZeroNET()
    #         for x,xc,gc in zip(self.global_model.parameters(True),self[cid].model.parameters(True),self.grads[cid].parameters(True)):
    #             gc.add_(x - xc)
        
    #     for params in zip(self.grad.parameters(True),*[self.grads[cid].parameters(True) for cid in self.grads.keys()]):
    #         params[0].fill_(0)
    #         for i in range(1,len(params)):
    #             params[0].add_(params[i])

    @torch.no_grad()
    def aggregate(self):
        """
            fedavg
        """
        self.update_grads()
        self.opt.zero_grad()
        for gm,grad in zip(self.global_model.parameters(True),self.grad.parameters(True)):
            # m.mul_(0.9).add_(grad,alpha=0.1)
            gm.grad = grad
        self.opt.step()
        
        return self.global_model
    