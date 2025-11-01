
import torch

from fedtorchPRO import Client
from fedtorchPRO.superconfig import SuperConfig

class DeltaSGDClient(Client):

    def traincore(self, arguments:SuperConfig, model, device):
        trainargs = arguments.client_trainer_arguments
        trainargs['opt'] = 'adsgd'
        trainer = arguments.TRAINER(
            model,
            train_dataset = self.train_dataset,
            test_dataset = None,
            device = device,
            **trainargs
        )
        return trainer.fit()