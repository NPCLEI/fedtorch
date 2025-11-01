
import sys
sys.path.append('./')

from fedtorchPRO.nn.instructor import EasyInstructor
from fedtorchPRO.nn.vit import VisionTransformer,vitsmall
from experiments.CIFAR100.libs.dataset import get_source,get_test
from experiments.CIFAR100.libs.nn import resnet101

trainer = EasyInstructor(resnet101(),get_source(),'cuda:0',100,opt='lamb',batch_size=1024,lr=1e-2,test_dataset=get_test())

trainer.fit()