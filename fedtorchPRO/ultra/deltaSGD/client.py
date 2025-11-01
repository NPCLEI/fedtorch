
import torch

from fedtorchPRO import Client
from .opt import deltaSGD
from fedtorchPRO.superconfig import SuperConfig

class DeltaSGD:
    def __init__(self,gamma):
        self.gamma = gamma

    def __call__(self,parameters, lr,weight_decay=None):

        return deltaSGD(parameters,lr,self.gamma)

class DeltaSGDClient(Client):

    def traincore(self, arguments:SuperConfig, model, device):
        trainer = arguments.TRAINER(
            model,
            train_dataset = self.train_dataset,
            test_dataset = None,
            device = device,
            **arguments.client_trainer_arguments
        )
        trainer.optc = DeltaSGD(arguments.get('deltasgd',0.01))
        return trainer.fit()