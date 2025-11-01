import time,torch

class Timer:
    """
    # 使用示例
    with Timer() as timer:
        total = 0
        for i in range(1000000):
            total += i
    print(f"Code executed in: {timer.elapsed_time:.4f} seconds")
    """
    def __enter__(self):
        self.start_time = time.time()
        return self  # 返回实例本身

    @property
    def cur_time(self):
        cur_time = time.time()
        return cur_time - self.start_time

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

    @staticmethod
    def interval(arr):
        if len(arr) <= 2:
            return sum(arr) / len(arr)

        max_val = max(arr)
        min_val = min(arr)
        
        # 去掉最大和最小值
        filtered_arr = [x for x in arr if x != max_val and x != min_val]
        
        return sum(filtered_arr) / len(filtered_arr)

class ModelTool:

    @staticmethod
    def cpu(md):
        return ModelTool.to(md,'cpu')

    @staticmethod
    def to(md,device):

        if md == None:
            return md

        if type(device) == str:
            device = torch.device(device)

        if hasattr(md,'device'):
            md_device = md.device
        else:
            for pm in md.parameters():
                md_device = pm.device
                break

        if md_device != device:
            if md_device == torch.device('cpu'):
                md.to(device)
            else:
                md.cpu().to(device)

        return md

    @staticmethod
    @torch.no_grad()
    def close_grad(net:torch.nn.Module):
        for pm in net.parameters():
            pm.requires_grad_(False)

    @staticmethod
    @torch.no_grad()
    def copynet(target,source):
        """
            target <- source
        """
        target.load_state_dict(source.state_dict())
        return target