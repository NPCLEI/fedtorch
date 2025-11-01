import torch
import torch.nn as nn
import multiprocessing
import time

class SharedMemoryCommunicatorGroup:
    def __init__(self, models):
        """
        初始化共享内存，并将每个模型的参数移到共享内存中
        参数:
        models: 包含多个模型的列表
        """
        self.shared_tensors_group = []
        for model in models:
            # 每个模型的参数列表保存在一个独立的 shared_tensors 中
            shared_tensors = [param.data.share_memory_() for param in model.parameters()]
            self.shared_tensors_group.append(shared_tensors)

    def update_model(self, model, model_index):
        """
        用共享内存中的张量更新指定索引的模型参数
        参数:
        model: 需要更新的模型
        model_index: 模型在列表中的索引
        """
        shared_tensors = self.shared_tensors_group[model_index]
        with torch.no_grad():
            for param, shared_tensor in zip(model.parameters(), shared_tensors):
                param.copy_(shared_tensor)

    def update_shared_tensors(self, model, model_index):
        """
        更新指定索引的模型的共享内存中的张量为模型的当前参数
        参数:
        model: 用于更新共享内存的模型
        model_index: 模型在列表中的索引
        """
        shared_tensors = self.shared_tensors_group[model_index]
        with torch.no_grad():
            for param, shared_tensor in zip(model.parameters(), shared_tensors):
                shared_tensor.copy_(param.data)

def randomize_model_parameters(model):
    """将模型的所有参数更新为随机值"""
    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-1, 1)  # 将每个参数更新为在 [-1, 1] 范围内的随机值

def model_sender(communicator_group, models, num_iterations):
    """发送模型的进程，反复更新共享内存中的模型"""
    for i in range(num_iterations):
        print(f"Iteration {i+1}: Randomizing and sending models in Process {multiprocessing.current_process().name}")
        # 随机化每个模型的参数，并更新共享内存
        for idx, model in enumerate(models):
            randomize_model_parameters(model)
            communicator_group.update_shared_tensors(model, idx)
        time.sleep(0.1)  # 模拟一些计算

def model_receiver(communicator_group, models, num_iterations):
    """接收模型的进程，反复从共享内存中更新模型"""
    for i in range(num_iterations):
        print(f"Iteration {i+1}: Receiving models in Process {multiprocessing.current_process().name}")
        # 从共享内存中读取每个模型的参数并更新模型
        for idx, model in enumerate(models):
            communicator_group.update_model(model, idx)
        time.sleep(0.1)  # 模拟一些计算

if __name__ == "__main__":
    # 创建多个简单的神经网络模型
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.struct1 = nn.Sequential(
                nn.Linear(28*28,1000,bias=False),
                nn.ReLU()
            )
            self.struct2 = nn.Linear(1000,62)

        def forward(self, x):
            x = x.reshape((x.size(0),-1))
            o = self.struct1(x)
            return self.struct2(o)

        def spms(self):
            return self.struct1.parameters()

    # 创建一个模型列表
    models = [SimpleNet() for i in range(100)]

    # 创建共享内存通信器组，将每个模型的参数移至共享内存
    communicator_group = SharedMemoryCommunicatorGroup(models)

    # 设定需要进行通讯的次数
    num_iterations = 10

    # 创建两个进程，一个负责发送模型，一个负责接收模型
    process1 = multiprocessing.Process(target=model_sender, args=(communicator_group, models, num_iterations))
    process2 = multiprocessing.Process(target=model_receiver, args=(communicator_group, models, num_iterations))
    process3 = multiprocessing.Process(target=model_receiver, args=(communicator_group, models, num_iterations))
    process4 = multiprocessing.Process(target=model_receiver, args=(communicator_group, models, num_iterations))

    # 启动两个进程
    process1.start()
    process2.start()
    process3.start()
    process4.start()

    # 等待两个进程结束
    process1.join()
    process2.join()
    process3.join()
    process4.join()

    print("Model communication test with shared memory and random tensor update for multiple models completed.")
