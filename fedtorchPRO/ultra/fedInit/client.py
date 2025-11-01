import torch

from fedtorchPRO.core.utils import ModelTool
from fedtorchPRO.core.client import Client
from fedtorchPRO.superconfig import SuperConfig

class FedInitClient(Client):

    def TrainModel(self, arguments: SuperConfig):
        device_id,model = arguments.acquire(self.cid)
        with torch.no_grad():
            if hasattr(self,'model_'):
                for pm,lm in zip(self.model_.parameters(True),model.parameters(True)):
                    pm.add_(lm - pm,alpha = arguments.fedinit)
        model,losses = self.traincore(arguments,model,device_id.split('/')[0])
        self.losse_record.clear()
        self.losse_record.extend(losses)
        with torch.no_grad():
            self.model = ModelTool.cpu(model)
            arguments.gpu_manager.release(device_id,self.cid)
            if arguments.keepnet:
                self.model_ = self.model_ if hasattr(self,'model_') else arguments.NET()
                for pm,om in zip(self.model_.parameters(True),self.model.parameters(True)):
                    pm.copy_(om)