
from .server import FedOPT

class FedMPX(FedOPT):

    def _init(self):
        super()._init()
        self.arguments.config['fedopt']['mpx'] = True
    