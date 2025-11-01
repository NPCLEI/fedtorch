import torch
from ..opt.server import FedOPT

# On the Role of Server Momentum in Federated Learning

class FedGM(FedOPT):

    @torch.no_grad()
    def _init(self):
        self._convert_FedAVG()
        self.d = self.arguments.NET()
        for pm in self.d.parameters(True):
            pm.fill_(0.0)
        self.fedgm = self.arguments.get('fedgm',(0.9,0.9))
        # self.arguments.weighted(close=True)
        return super()._init()

    @torch.no_grad()
    def aggregate(self):
        """
            fedavg
        """

        weight = self.weight
        betas,vs = self.fedgm

        self.opt.zero_grad()

        for params in zip(
                self.global_model.parameters(True),self.d.parameters(True),
                *[self[cid].model.parameters(True) for cid in self.selected_ids]
            ):
            grad = (params[0] - params[2]) * weight[0]
            for idx in range(3,len(params)):
                grad.add_(params[0] - params[idx],alpha = weight[idx-2])
            params[1].copy_((1-betas)  * grad + betas * params[1])
            params[0].grad = (1-vs) * grad + vs * params[1]

        self.opt.step()

        return self.global_model