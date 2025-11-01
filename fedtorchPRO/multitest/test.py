import torch
import torch.nn as nn
import multiprocessing
import time

class SharedMemoryCommunicator:
    def __init__(self, model):
        """初始化共享内存，并将模型的参数移到共享内存中"""
        self.shared_tensors = [param.data.share_memory_() for param in model.parameters()]
    
    def get_shared_tensors(self):
        """获取共享内存中的张量"""
        return self.shared_tensors
    
    def update_model(self, model):
        """用共享内存中的张量更新模型参数"""
        with torch.no_grad():
            for param, shared_tensor in zip(model.parameters(), self.shared_tensors):
                param.copy_(shared_tensor)

    def update_shared_tensors(self, model):
        """更新共享内存中的张量为模型的当前参数"""
        with torch.no_grad():
            for param, shared_tensor in zip(model.parameters(), self.shared_tensors):
                shared_tensor.copy_(param.data)

def randomize_model_parameters(model):
    """将模型的所有参数更新为随机值"""
    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-1, 1)  # 将每个参数更新为在 [-1, 1] 范围内的随机值

def model_sender(communicator, model, num_iterations):
    """发送模型的进程，反复更新共享内存中的模型"""
    for i in range(num_iterations):
        print(f"Iteration {i+1}: Randomizing and sending model in Process {multiprocessing.current_process().name}")
        # 先随机化模型参数
        randomize_model_parameters(model)
        # 将随机化后的参数更新到共享内存中
        communicator.update_shared_tensors(model)
        time.sleep(0.1)  # 模拟一些计算

def model_receiver(communicator, model, num_iterations):
    """接收模型的进程，反复从共享内存中更新模型"""
    for i in range(num_iterations):
        print(f"Iteration {i+1}: Receiving model in Process {multiprocessing.current_process().name}")
        # 从共享内存中读取参数更新模型
        communicator.update_model(model)
        time.sleep(0.1)  # 模拟一些计算

if __name__ == "__main__":
    # 创建一个简单的神经网络模型
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

    model = SimpleNet()

    # 创建共享内存通信器，将模型参数移至共享内存
    communicator = SharedMemoryCommunicator(model)

    # 设定需要进行通讯的次数
    num_iterations = 10

    model_sender(communicator,model,1)

    # 创建两个进程，一个负责发送模型，一个负责接收模型
    # process1 = multiprocessing.Process(target=model_sender, args=(communicator, model, num_iterations))
    process2 = multiprocessing.Process(target=model_receiver, args=(communicator, model, num_iterations))
    process3 = multiprocessing.Process(target=model_receiver, args=(communicator, model, num_iterations))
    process4 = multiprocessing.Process(target=model_receiver, args=(communicator, model, num_iterations))

    # 启动两个进程
    # process1.start()
    process2.start()
    process3.start()
    process4.start()

    # 等待两个进程结束
    # process1.join()
    process2.join()
    process3.join()
    process4.join()

    print("Model communication test with shared memory and random tensor update completed.")
