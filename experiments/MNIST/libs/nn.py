
from torch import nn

class logicregress(nn.Module):
    def __init__(self):
        super(logicregress, self).__init__()
        self.struct1 = nn.Sequential(
            nn.Linear(28*28,1000,bias=False),
            nn.ReLU()
        )
        self.struct2 = nn.Linear(1000,10)

    def forward(self, x):
        x = x.reshape((x.size(0),-1))
        o = self.struct1(x)
        return self.struct2(o)

    def spms(self):
        return self.struct1.parameters()

from fedtorchPRO.nn.instructor import EasyInstructor

def net(netname):
    return logicregress()

NET = net
TRAINER = EasyInstructor



