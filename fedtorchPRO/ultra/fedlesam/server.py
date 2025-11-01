
from .client import FedLESAMClient

from fedtorchPRO.ultra.opt.server import FedOPT
from fedtorchPRO.core.cache import ModelManager

# Title: Locally Estimated Global Perturbations are Better than Local Perturbations for Federated Sharpness-aware Minimization.

class FedLESAM(FedOPT):

    def _init(self):
        # 强制变成FedAVG
        self._convert_FedAVG()
        self.CLIENT = FedLESAMClient
        self.wt_manager = ModelManager(self.arguments.NET)
        super()._init()
