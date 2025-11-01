
import torch.nn as nn
import torch.nn.functional as F

class logical(nn.Module):
    def __init__(self):
        super(logical, self).__init__()
        self.feature = nn.Linear(28*28,1000,bias=False)
        self.cls = nn.Linear(1000,62,bias=False)

    def forward(self, x):
        x = x.reshape((x.size(0),-1))
        x = F.relu(self.feature(x))
        return self.cls(x)

class MLPMNISTCR(nn.Module):
    def __init__(self):
        super(MLPMNISTCR, self).__init__()
        self.struct1 = nn.Sequential(
            nn.Linear(28*28,100,bias=False),
            nn.LayerNorm(100),
            nn.ReLU()
        )
        self.struct2 = nn.Sequential(
            nn.Linear(100,100,bias=False),
            nn.LayerNorm(100),
            nn.ReLU()
        )
        self.struct3 = nn.Sequential(
            nn.Linear(100,100,bias=False),
            nn.LayerNorm(100),
            nn.ReLU()
        )
        self.struct4 = nn.Sequential(
            nn.Linear(100,100,bias=False),
            nn.LayerNorm(100),
            nn.ReLU()
        )
        self.cls = nn.Linear(100,62,bias=False)

    def forward(self, x):
        x = x.reshape((x.size(0),-1))
        o = self.struct1(x)
        o = self.struct2(o)
        o = self.struct3(o)
        o = self.struct4(o)
        return self.cls(o)

    def spms(self):
        return self.struct1.parameters()


# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.group_norm1 = nn.GroupNorm(16, 32)  # Group normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(32, 64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 62)  # 26 classes for EMNIST ByClass

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
from .restnet import resnet1,resnet6,resnet12,resnet18

netmap = {
    'resnet18':resnet18,
    'resnet12':resnet12,
    'resnet6' :resnet6,
    'resnet1' :resnet1,
    'fullconnect':MLPMNISTCR,
    'cnn':CNN,
    'logical':logical
}

def net(netname):
    return netmap[netname]()

NET = net
TRAINER = EasyInstructor


