import torch,os

from fedtorchPRO.core.server import Server

class ADCure(Server):

    def _init(self):
        super()._init()
        self.global_model.load_state_dict(torch.load(self.arguments.x_start+'/model.states.pth',weights_only=True))

    @torch.no_grad()
    def aggregate(self):

        x_start = self.arguments.inimodel()
        x_start.load_state_dict(torch.load(self.arguments.x_start+'/x_start.pth',weights_only=True))
        ds = []
        for params in zip(x_start.parameters(True),*[self[cid].model.parameters(True) for cid in self.selected_ids]):
            d = (params[0] -  params[1]).norm()
            for idx in range(2,len(params)):
                d = d + (params[0] -  params[idx]).norm()
            ds.append(d.item())
        
        self.DV = sum(ds)/len(ds)

        os.makedirs(self.arguments.run_path,exist_ok=True)
        method_path = os.path.join(self.arguments.run_path,self.arguments.methodname)
        os.makedirs(method_path,exist_ok=True)
        with open(os.path.join(method_path,'ADC_V.txt'),'w+',encoding='utf8') as fp:
            fp.write(str(self.DV))
        for cid in self.selected_ids:
            torch.save(self[cid].model.state_dict(),os.path.join(method_path,f'model_{cid}'))

    def save(self):
        import os

        print()
        print(self.log_title,"SAVED.")
        print()