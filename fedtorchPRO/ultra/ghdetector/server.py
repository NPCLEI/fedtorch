import torch

from .client import GDHClient
from fedtorchPRO import FedSGD
from fedtorchPRO.core.cache import GPUManager,ModelManager

class GHDetector(FedSGD):

    def _init(self):
        super()._init()
        self.gpu_manager = GPUManager(self.arguments.available_gpus)
        self.mdl_manager = ModelManager(self.arguments.NET)
        self.heter_record = []
        self.CLIENT = GDHClient

    @torch.no_grad()
    def acquire(self,cid):
        did = self.gpu_manager.acquire(cid)
        model = self.mdl_manager.acquire(self.selected_ids.index(cid))
        for target,source in zip(model.parameters(),self.global_model.parameters()):
            target.copy_(source)
        return did,model,self.arguments.get('batch_size')

    def before_train(self):
        # for cid in self.selected_ids:
        #     self.clients[cid].fullbatchgradi(self)
        # self.compute_gradi_heter()
        # return
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.available_gpus)) as executor:
            futures = {executor.submit(self.clients[cid].fullbatchgradi, self): cid for cid in self.selected_ids}
            for future in concurrent.futures.as_completed(futures):
                tid = futures[future]
                try:
                    future.result()  # 获取线程结果，确保异常被捕获
                except Exception as e:
                    print(f"Worker {tid} raised an exception: {e}")
        self.compute_gradi_heter()

    @torch.no_grad()
    def compute_gradi_heter(self):
        global_gradi = self.mdl_manager.acquire(-1)
        for pms in zip(global_gradi.parameters(),*[self[cid].gradi.parameters() for cid in self.selected_ids]):
            gg,gis = pms[0],pms[1:]
            gg.copy_(gis[0])
            for i in range(2,len(gis)):
                gg.add_(gis[i-1])
            gg.div_(1/len(self.selected_ids))
        heter = []
        for pms in zip(global_gradi.parameters(),*[self[cid].gradi.parameters() for cid in self.selected_ids]):
            gg,gis = pms[0],pms[1:]
            hi = 0
            for i in range(len(gis)):
                hi += (gg - gis[i]).norm()
            heter.append(hi / len(gis))
        self.heter_record.append((self.cur_comic,torch.tensor(heter).mean().item()))

    def train(self):
        self.before_train()
        return super().train()
    
    def save(self):
        self._save_lst(self.heter_record,'het')
        return super().save()