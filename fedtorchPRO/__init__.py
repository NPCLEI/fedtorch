# fedtorchPRO/__init__.py

import os

# 设置环境变量
os.environ['PYTHON_GIL'] = '0'

from .exp import exp
from .core.server import Server as FedAVG
from .core.server import Server
from .core.client import Client
from .core.utils import ModelTool,ModelTool
from .superconfig import SuperConfig

from .ultra.opt import FedOPT,FedMPX,FedSGDM,FedAdagrad,FedADAM,FedSGD,FedAVGPlus,FedAMS
from .ultra.prox.client import ProxClient
from .ultra.deltaSGD.client import DeltaSGDClient
from .ultra.exp.server import FedEXP
from .ultra.dualproxm.client import DualProxMClient
from .ultra.fedInit import FedInit,FedInitClient
from .ultra.fedlesam import FedLESAM
from .ultra.fedgm.server import FedGM
from .ultra.fednar.client import FedNARClient
from .ultra.scaffold.server import SCAFFOLD
from .ultra.scaffnew.server import SCAFFNEW
from .ultra.avgM.server import FedAVGM
from .ultra.ours.server import Ours
from .ultra.ours.server_adp import OursADP
from .ultra.gradi_cos_analyse.server import GradiAnalyseServer
from .ultra.adcurve.server import ADCure
from .ultra.ghdetector.server import GHDetector
from .ultra.visual.process import VisPrc 

FedPROX = (FedAVGPlus,ProxClient)
DeltaSGD = (FedAVGPlus,DeltaSGDClient)
DualPROX = (FedAVGPlus,DualProxMClient)
FedNAR = (FedAVGPlus,FedNARClient)
