from typing import Any
import torch

from fedtorchPRO.core.utils import ModelTool
from fedtorchPRO.core.client import Client
from fedtorchPRO.superconfig import SuperConfig

class lesam_ontrain_handle:

    def __init__(self,rho,old_w,wt,device) -> None:
        self.rho = rho
        self.old_w = old_w
        self.wt = wt
        ModelTool.to(wt,device)
        ModelTool.to(old_w,device)

    # @torch.no_grad()
    # def __call__(self, net) -> Any:
    #     for pm,om,wt in zip(net.parameters(),self.old_w.parameters(),self.wt.parameters()):
    #         delta = om - wt
    #         pm.add_(delta/(delta.norm() + 1e-4),alpha = self.rho)


    @torch.no_grad()
    def __call__(self, net) -> Any:
        sum_norm = 0
        for pm,om,wt in zip(net.parameters(),self.old_w.parameters(),self.wt.parameters()):
            sum_norm += (om - wt).norm()
        scale = self.rho / (sum_norm + 1e-7)
        for pm,om,wt in zip(net.parameters(),self.old_w.parameters(),self.wt.parameters()):
            delta = om - wt
            pm.add_(delta ,alpha = scale)

    def cpu(self):
        ModelTool.cpu(self.wt)
        ModelTool.cpu(self.old_w)

class FedLESAMClient(Client):

    def traincore(self, arguments:SuperConfig, model, device):
        model,wt = model
        trainer = arguments.TRAINER(
            model,
            train_dataset = self.train_dataset,
            test_dataset = None,
            device = device,
            **arguments.client_trainer_arguments
        )
        handler = lesam_ontrain_handle(arguments.get('fedlesam',0.01),self.old_w,wt,device)
        trainer.ontrainhandle = handler
        res = trainer.fit()
        handler.cpu()
        return res

    def TrainModel(self,arguments:SuperConfig):
        with torch.no_grad():
            self.old_w = self.old_w if hasattr(self,'old_w') else arguments.NET()
            device_id,model = arguments.acquire(self.cid)
            wt = arguments.server.wt_manager.acquire(arguments.server.selected_ids.index(self.cid),model)
 
        model,losses = self.traincore(arguments,(model,wt),device_id.split('/')[0])
        self.losse_record.clear()
        self.losse_record.extend(losses)
        arguments.gpu_manager.release(device_id,self.cid)

        self.model  = ModelTool.cpu(model)
        self.old_w = ModelTool.cpu(self.old_w)
        ModelTool.cpu(wt)
        with torch.no_grad():
            self.old_w.load_state_dict(wt.state_dict())
            