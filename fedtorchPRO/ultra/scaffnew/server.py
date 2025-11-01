
from ..opt.servers import FedAVGPlus
from .client import SCAFFNEWClient
from fedtorchPRO.core.cache import ModelManager

class SCAFFNEW(FedAVGPlus):
    def _init(self):
        self.glbs = ModelManager(self.arguments.NET)
        self.CLIENT = SCAFFNEWClient
        return super()._init()

    def acquire_global_model(self,cid):
        return self.glbs.acquire(self.selected_ids.index(cid),self.global_model)