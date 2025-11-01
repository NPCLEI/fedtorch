
import torch

from fedtorchPRO.core.utils import ModelTool
from fedtorchPRO.superconfig import SuperConfig

class Client:

    def __init__(self,train_dataset,cid) -> None:
        self.train_dataset = train_dataset
        self.nk = len(train_dataset)
        self.cid = cid
        self.losse_record = []
        self.net = None
        self._init()

    def _init(self):
        return

    def traincore(self,arguments,model,device):
        trainer = arguments.TRAINER(
            model,
            train_dataset = self.train_dataset,
            test_dataset = None,
            device = device,
            **arguments.client_trainer_arguments
        )
        return trainer.fit()

    def TrainModel(self,arguments:SuperConfig):
        # self.to(self.device)
        device_id,model = arguments.acquire(self.cid)
        batch_size = arguments.batch_size
        bp = batch_size
        while bp > 1:
            try:
                model,losses = self.traincore(arguments,model,device_id.split('/')[0])
                arguments.client_trainer_arguments['batch_size'] = batch_size
                break  # 如果没有发生 CUDA OOM 错误，则退出循环
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"CUDA out of memory, reducing batch size from {batch_size} to {batch_size * 0.9}")
                    bp = int(bp * 0.8)
                    arguments.client_trainer_arguments['batch_size'] = bp
                    torch.cuda.empty_cache()  # 清理CUDA缓存
                else:
                    raise  # 如果不是CUDA内存问题，重新抛出异常
        self.losse_record.clear()
        self.losse_record.extend(losses)
        self.model = ModelTool.cpu(model)
        arguments.gpu_manager.release(device_id,self.cid)
        if arguments.keepnet:
            self.model_ = self.model_ if hasattr(self,'model_') else arguments.NET()
            self.model_ = ModelTool.copynet(self.model_,model)
