
from .client import FedInitClient

from fedtorchPRO.superconfig import SuperConfig
from fedtorchPRO.ultra.opt.server import FedOPT

class FedInit(FedOPT):

    def _init(self):
        self._convert_FedAVG()
        super()._init()
        self.arguments.config['workconfig']['keepnet'] = True
