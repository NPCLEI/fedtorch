
import math
import torch

from ...core.client import Client
from fedtorchPRO import ModelTool,SuperConfig

class scaffold_gradhandle:

    def __init__(self,ci,c,device) -> None:
        ModelTool.to(ci,device)
        ModelTool.to(c,device)
        self.ci = ci
        self.c = c

    @torch.no_grad()
    def __call__(self,net):
        for g,ci,c in zip(net.parameters(),self.ci.parameters(),self.c.parameters()):
            g.grad = g.grad - ci + c

    def cpu(self):
        ModelTool.cpu(self.ci)
        ModelTool.cpu(self.c)

class SCAFFOLDClient(Client):

    def _init(self):
        self.inited = False

    def _init_param(self,args : SuperConfig):
        if self.inited:
            return
        self.ci = args.ZeroNET()
        self.ci_plus = args.ZeroNET()
        self.ci_delta = args.ZeroNET()
        self.inited = True

    @torch.no_grad()
    def set_new_controls_option_II(self,args:SuperConfig,c,x0,xt):
        cta = args.client_trainer_arguments
        E,LR,B = cta['epoch'],cta['lr'],cta['batch_size']
        for cip,ci,c,xi,x0,deltaci in zip(
            self.ci_plus.parameters(True),
            self.ci.parameters(True),c.parameters(True),
            xt.parameters(True),x0.parameters(True),
            self.ci_delta.parameters(True)
        ):
            cip.copy_(ci - c + ((E*LR*B)**(-1)) * (xi - x0))
            deltaci.copy_(cip - ci)
            ci.copy_(cip)

    def traincore(self,args:SuperConfig,model,device):
        self._init_param(args)
        trainer = args.TRAINER(
            model,
            train_dataset = self.train_dataset,
            test_dataset = None,
            device = device,
            **args.client_trainer_arguments
        )
        server_control = args.server.aquiresc(self.cid)
        trainer.gradhandle = scaffold_gradhandle(self.ci,server_control,device)
        res = trainer.fit()
        trainer.gradhandle.cpu()
        self.set_new_controls_option_II(args,server_control,args.server.aquiregm(self.cid),res[0])
        return res