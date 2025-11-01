

import sys
sys.path.append('./')

import torch

from fedtorchPRO.imitate import transforms 
from fedtorchPRO.superconfig import SuperConfig
from fedtorchPRO.nn.instructor import EasyInstructor
from fedtorchPRO.nn.vit import VisionTransformer,vitsmall

from experiments.CIFAR100.libs.restnet import resnet18,resnet50,resnet101

NETMAP = {
    "resnet18":resnet18,
    "resnet50":resnet50,
    "resnet101":resnet101,
    "vitbase":VisionTransformer,
    "vitsmall":vitsmall
}

def ChooseNet(netname):
    return NETMAP[netname](num_classes = 100)

NET = ChooseNet
TRAINER = EasyInstructor

