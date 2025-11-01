
import torch

from fedtorchPRO import Client
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class NARLossf:
    def __init__(self,net,l2_reg,lossf):
        self.net = net
        self.l2 = l2_reg
        self.lossf = lossf

    def __call__(self,output, target):
        loss = self.lossf(output ,target)
        if self.l2 == 0:
            return loss
        # local_par_list = parameters_to_vector(self.net.parameters())
        # return loss + 0.5 * self.l2 * torch.norm(local_par_list) ** 2
        for pm in self.net.parameters():
            loss = loss + 0.5 * self.l2 * pm.norm(2).pow(2)
        return loss
    
class FedNARClient(Client):

    def traincore(self, arguments, model, device):
        trainer = arguments.TRAINER(
            model,
            train_dataset = self.train_dataset,
            test_dataset = None,
            device = device,
            **arguments.client_trainer_arguments
        )
        lossf = NARLossf(model,arguments.get('fednar',0.001),trainer.lossf())
        trainer.lossf = lambda : lossf
        return trainer.fit()