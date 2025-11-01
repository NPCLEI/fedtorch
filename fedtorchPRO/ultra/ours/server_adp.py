

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
from fedtorchPRO.superconfig import Parser

class OursADP(FedOPT):

    def _init(self):
        self.arguments.config['fedopt']['betas'] = (0.,0.999)
        self.arguments.config['fedopt']['method'] = 'sgd'
        self.arguments.config['fedopt']['momentum'] = 0
        self.v = self.arguments.ZeroNET()
        reparser = Parser(self.arguments.config)
        reparser(self.arguments.command)
        self.arguments.config = reparser.config
        super()._init()

    def _onrun(self):
        super()._onrun()
        self.grads = {}
        self.grdst = [0 for _ in range(len(self))]
        self.grad = self.arguments.ZeroNET()
        # self.mt = self.arguments.ZeroNET()
        self.beta1 = self.arguments.get('ours_beta',0.9)
        self.lmbda = self.arguments.get('ours_grad_decay',0.99)
        self.lmbda2 = 0.999
        self.total_weights = self.weights(self.client_ids)

    @torch.no_grad()
    def update_grads(self):
        # Perfect !

        for id,cid in enumerate(self.selected_ids):
            for x,xc,gc in zip(self.global_model.parameters(True),self[cid].model.parameters(True),self.grad.parameters(True)):
                gc.mul_(self.lmbda)
                grad = x - xc
                self.lr_adp = grad.norm()
                gc.add_(grad)
        
        for gt,vt in zip(self.grad.parameters(True),self.v.parameters(True)):
            vt.mul_(self.lmbda2)
            vt.add_(gt.pow(2)) 

    @torch.no_grad()
    def aggregate(self):
        """
            fedavg
        """
        self.update_grads()
        self.opt.zero_grad()
        for gm,grad,v in zip(self.global_model.parameters(True),self.grad.parameters(True),self.v.parameters(True)):
            gm.grad = grad / (v.mul(1/(1 - self.lmbda2**(self.cur_comic+1))).sqrt() + 1e-4) 
        self.opt.step()
        
        return self.global_model

    # @torch.no_grad()
    # def update_grads_(self):
    #     """
    #         It worked ! but not perfect.
    #     """
    #     for id,cid in enumerate(self.selected_ids):
    #         if cid not in self.grads:
    #             self.grads[cid] = self.arguments.NET()
    #         for x,xc,gc in zip(self.global_model.parameters(True),self[cid].model.parameters(True),self.grads[cid].parameters(True)):
    #             gc.copy_(x - xc)
    #         self.grdst[cid] = self.cur_comic
        
    #     weight = self.weight
    #     for params in zip(self.grad.parameters(True),*[self.grads[cid].parameters(True) for cid in self.selected_ids]):
    #         params[0].fill_(0)
    #         for i in range(1,len(params)):
    #             params[0].add_(params[i],alpha= weight[i -1])

    #     if self.cur_comic < 1:
    #         return
        
    #     keys = [cid for cid in self.grads.keys() if cid not in self.selected_ids]
    #     weight = self.weights(keys)
    #     for params in zip(self.grad.parameters(True),*[self.grads[cid].parameters(True) for cid in keys]):
    #         # params[0].mul_(self.p)
    #         for i in range(1,len(params)):
    #             params[0].add_(params[i],alpha= (1 - self.p) * weight[i-1] * self.lmbda ** (self.cur_comic - self.grdst[keys[i - 1]]))

    # @torch.no_grad()
    # def update_grads(self):
    #     # Perfect !
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