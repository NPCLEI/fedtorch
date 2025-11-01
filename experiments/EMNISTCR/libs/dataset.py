
import torch

from fedtorchPRO.imitate.datasets.mnist import EMNIST
from fedtorchPRO.imitate import transforms 

class Div255(torch.nn.Module):
    def forward(self,x:torch.Tensor):
        return x.div_(255.0)

def get_source():
    return EMNIST( 
        root='./datasets/EMNIST/',
        train=True,
        download=False,
        transform = transforms.Compose([
            transforms.ToTensor(),
            Div255()
        ]),
        split='byclass',
    )

def get_test():
    return EMNIST( 
        root='./datasets/EMNIST/',
        train=False,
        download=False,
        transform = transforms.Compose([
            transforms.ToTensor(),
            Div255()
        ]),
        split='byclass' 
    )
