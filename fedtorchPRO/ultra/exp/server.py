import torch
from fedtorchPRO import FedAVGPlus

class FedEXP(FedAVGPlus):

    def _init(self):
        self.arguments.weighted(False)
        return super()._init()

    @torch.no_grad()
    def aggregate(self):

        for params in zip(
                self.global_model.parameters(True),
                *[self[cid].model.parameters(True) for cid in self.selected_ids]
            ):
            grad = params[0] - params[1]
            eta = grad.norm(2).pow(2)
            grad = grad * self.weight[0]
            for idx in range(2,len(params)):
                gradi = params[0] - params[idx]
                grad.add_(gradi,alpha = self.weight[idx-1])
                ########exp##############
                eta += gradi.norm(2).pow(2)
            eta = max(1, eta / (2 * len(self.selected_ids) * (grad.norm(2).pow(2) + 1e-8)))
            params[0].data.add_(grad,alpha = -eta)

        return self.global_model
