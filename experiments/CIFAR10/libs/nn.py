

import sys
sys.path.append('./')

from fedtorchPRO.nn.instructor import EasyInstructor
from fedtorchPRO.nn.vit import VisionTransformer,vitsmall

from experiments.CIFAR100.libs.restnet import resnet18,resnet50,resnet101,resnet34

NETMAP = {
    "resnet18":resnet18,
    "resnet50":resnet50,
    "resnet34":resnet34,
    "resnet101":resnet101,
    "vitbase":VisionTransformer,
    "vitsmall":vitsmall
}

def ChooseNet(netname):
    return NETMAP[netname]()

NET = ChooseNet
TRAINER = EasyInstructor

