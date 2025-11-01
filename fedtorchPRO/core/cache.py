import threading,torch

PRINTINFO = False

class ModelManager:
    def __init__(self, ininet):
        self.NET = ininet
        self.nets = {}
        self.lock = threading.Lock()

    def acquire(self, id , source = None) -> torch.nn.Module:
        with self.lock:
            if id not in self.nets:
                self.nets[id] = self.NET()
            if source:
                self.nets[id].load_state_dict(source.state_dict())
            return self.nets[id]

class GPUManager:

    def __init__(self, available_gpus):
        """
        初始化 GPUManager，接收可用 GPU 列表，类似 ['cuda:0', 'cuda:3', 'cuda:9'].
        """
        import random
        random.shuffle(available_gpus)
        self._available_gpus = available_gpus
        self.gpus = {gpu: None for gpu in available_gpus}  # 用字典记录 GPU 的状态 (None 表示可用)
        self.lock = threading.Lock()  # 用于控制访问 gpus 字典的锁

    def acquire(self, id):
        """
        尝试申请一个可用的 GPU 资源，返回申请的 GPU 标识符.
        """
        with self.lock:
            for gpu, owner in self.gpus.items():
                if owner is None:  # 如果找到可用的 GPU
                    self.gpus[gpu] = id  # 记录该 GPU 被哪个 id 使用
                    if PRINTINFO:
                        print(f"{gpu} is acquired by {id}")
                    return gpu  # 返回申请的 GPU 标识符
            raise RuntimeError("No GPUs available")

    def release(self, gpu, id):
        """
        释放指定的 GPU 资源，检查释放的 ID 是否与申请时一致.
        """
        with self.lock:
            if gpu not in self.gpus:
                raise ValueError(f"{gpu} is not a valid GPU identifier")
            if self.gpus[gpu] == id:
                self.gpus[gpu] = None  # 释放 GPU
                if PRINTINFO:
                    print(f"{gpu} is released by {id}")
            else:
                raise RuntimeError(f"{gpu} is not acquired by {id}, cannot release")

