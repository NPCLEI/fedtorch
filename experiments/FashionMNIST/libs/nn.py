import sys
sys.path.append('./')

from torch import nn

class MLPMNISTCR(nn.Module):
    def __init__(self):
        super(MLPMNISTCR, self).__init__()
        self.struct1 = nn.Sequential(
            nn.Linear(28*28,1000,bias=False),
            nn.ReLU()
        )
        self.struct2 = nn.Linear(1000,62)

    def forward(self, x):
        x = x.reshape((x.size(0),-1))
        o = self.struct1(x)
        return self.struct2(o)

    def spms(self):
        return self.struct1.parameters()

import torch.nn as nn
import torch.nn.functional as F

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self,num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.group_norm1 = nn.GroupNorm(8, 32)  # Group normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(16, 64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)  # 26 classes for EMNIST ByClass

    def forward(self, x):
        x = F.relu(self.group_norm1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.group_norm2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

from fedtorchPRO.nn.instructor import EasyInstructor

from experiments.FashionMNIST.libs.restnet import resnet18,resnet50,resnet101,resnet34

NETMAP = {
    "resnet18":resnet18,
    "resnet50":resnet50,
    "resnet34":resnet34,
    "resnet101":resnet101,
    "cnn":CNN
}

def ChooseNet(netname):
    return NETMAP[netname](num_classes = 10)

NET = ChooseNet
TRAINER = EasyInstructor