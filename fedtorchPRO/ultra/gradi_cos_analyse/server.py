import torch
from fedtorchPRO.core.server import Server

class GradiAnalyseServer(Server):

    def _init(self):
        super()._init()
        self.layer_grai_ang_record = []
        self.full_client_gradi = self.arguments.inimodel()
    
    def _onrun(self):
        super()._onrun()
        self.opt = torch.optim.Adam(self.global_model.parameters(),self.arguments.server_lr)
    
    def select(self):
        super().select()
        self.orselect = [i for i in self.selected_ids]
        self.selected_ids = self.client_ids
        self.lr = self.arguments.server_lr

    @torch.no_grad()
    def aggregate(self):
        """
            fedavg
        """

        for params in zip(
                self.full_client_gradi.parameters(True),
                *[self[cid].model.parameters(True) for cid in self.selected_ids]
            ):
            grad = (params[0].detach() - params[1])
            for idx in range(2,len(params)):
                grad.add_(params[0].detach() - params[idx])
            params[0].grad = grad

        self.opt.zero_grad()        
        for params in zip(
                self.global_model.parameters(True),
                *[self[cid].model.parameters(True) for cid in self.orselect]
            ):
            grad = (params[0].detach() - params[1])
            for idx in range(2,len(params)):
                grad.add_(params[0].detach() - params[idx])
            params[0].grad = grad
        self.opt.step()

        angles = []
        for ful,glb in zip(self.full_client_gradi.parameters(True),self.global_model.parameters(True)):
            angles.append(torch.arccos(torch.cosine_similarity(ful.grad.flatten(),glb.grad.flatten(),dim=0)).mean())
            ful.copy_(glb)
            
        for i,angle in enumerate(angles):      
            self.recorder.log_metric('gradi_angle',self.cur_comic,angle,metric_type='ang.ly.%d'%i)

        return self.global_model