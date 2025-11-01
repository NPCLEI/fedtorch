import os,torch

from fedtorchPRO.core.server import Server
from fedtorchPRO.core.utils import ModelTool
from fedtorchPRO.data.utils import DataShell

class VisPrc(Server):

    def _tempdata(self):
        res = []
        for cid in self.selected_ids:
            res.extend(self[cid].train_dataset.mapidx)
        return DataShell(res)

    def save_traj(self):
        os.makedirs(self.arguments.run_path,exist_ok=True)
        method_path = os.path.join(self.arguments.run_path,self.arguments.methodname)
        os.makedirs(method_path,exist_ok=True)
        save_path = os.path.join(method_path,'round%s'%self.cur_comic)
        os.makedirs(save_path,exist_ok=True)
        for cid in self.selected_ids:
            torch.save(self[cid].model.state_dict(), os.path.join(save_path ,f'client.{cid}.model.states.pth'))
        
        torch.save(self.global_model.state_dict(), os.path.join(save_path ,'fedavg.model.states.pth'))
        glb = self.train_to_optimal()
        torch.save(glb.state_dict(), os.path.join(save_path ,'global.model.states.pth'))

        print("VP report : saved. path:",save_path)

    def _init(self):
        super()._init()
        self.to_optimal_config = {
            "epoch" : 20,
            "lr" : 1e-2,
            "batch_size":1024,
            "opt":"adam",
            "util_loss":1e-1
        }   

    def train_to_optimal(self):
        model = ModelTool.copynet(self.arguments.NET(),self.global_model)
        trainer = self.arguments.TRAINER(
            model,
            train_dataset = self._tempdata(),
            test_dataset = None,
            device = self.arguments.gpu_manager._available_gpus[0].split('/')[0],
            **self.to_optimal_config
        )
        net,losses =  trainer.fit()
        return net

    def aggregate(self):
        """
            局部最优点和全局最优点的距离随t时刻改变而改变。
        """
        super().aggregate()
        self.save_traj()