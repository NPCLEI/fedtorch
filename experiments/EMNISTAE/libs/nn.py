

import sys
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from fedtorchPRO.nn.instructor import set_seed

class AEL(nn.Module):
    def __init__(self):
        super(AEL, self).__init__()
        # Encoder部分
        self.encoder = nn.Sequential(
            nn.Linear(28*28,512),nn.LayerNorm(512),nn.Sigmoid(),
            nn.Linear(512,256),nn.LayerNorm(256),nn.ReLU(),
            nn.Linear(256,64),nn.LayerNorm(64),nn.ReLU(),
            nn.Linear(64,16),nn.Sigmoid(),
        )
        
        # Decoder部分
        self.decoder = nn.Sequential(
            nn.Linear(16,64),nn.LayerNorm(64),nn.ReLU(),
            nn.Linear(64,256),nn.LayerNorm(256),nn.ReLU(),
            nn.Linear(256,512),nn.LayerNorm(512),nn.Sigmoid(),
            nn.Linear(512,28*28)
        )

    def forward(self, x):
        x = self.encoder(x.flatten(1))
        x = self.decoder(x)
        return x.view(-1,1,28,28)
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder部分
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输入：1x28x28，输出：32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 输出：32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 输出：64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                           # 输出：64x7x7
        )
        
        # Decoder部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # 输出：32x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),  # 输出：1x28x28
            nn.Tanh()                                            # 将输出限制在[-1,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

from fedtorchPRO.nn.instructor import EasyInstructor

class AutoEncoderInstructor(EasyInstructor):
    def __init__(self, net, train_dataset, device, **kwargs):
        # 调用父类的构造函数
        super().__init__(net, train_dataset, device, **kwargs)
        # 默认使用MSE
        self.selectlossf('mse')
        

    @staticmethod
    def lossfn(net, batch, device, lossf=None):
        """
        重载损失函数，用于AutoEncoder任务
        """
        if lossf is None:
            lossf = torch.nn.functional.mse_loss  # 默认使用MSE
        output = net(batch[0].to(device))
        loss = lossf(output, batch[0].to(device))  # 重构目标
        return loss, output

    @staticmethod
    def TestModel(net,test_dataset,batch_size = 2048,device = 'cuda:0',commic_times = -1):
        return AutoEncoderInstructor._test_model_(net,test_dataset,batch_size,device,commic_times)
    
    @staticmethod
    @torch.no_grad()
    def _test_model_(net,test_dataset,batch_size = 64,device = 'cuda:0',commic_times = -1,cache = None,num_workers = 2):
        """
        重载测试方法，用于AutoEncoder任务
        """
        set_seed()

        if test_dataset is None:
            if cache is not None:
                cache.append(-1)
                cache.append(None)
            return -1, None

        # total_loss = 0
        # total_samples = 0
        mean_loss = 0
        AutoEncoderInstructor.to(net, device)
        loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)

        for batch in tqdm(loader):
            inputs = batch[0].to(device)  # AutoEncoder 任务只关注输入
        
            # 通过 AutoEncoder 进行前向传播并计算损失
            loss, outputs = AutoEncoderInstructor.lossfn(net, batch, device)  # 使用重载的损失函数

            # total_loss += loss.item() * inputs.size(0)  # 累加总损失（使用样本数加权）
            # total_samples += inputs.size(0)  # 统计样本数
            mean_loss += loss.detach().item()
            
        mean_loss = mean_loss / (len(loader) + 1e-8)

        # mean_loss = total_loss / total_samples if total_samples > 0 else float('inf')  # 计算平均损失

        print("测试损失",mean_loss)

        AutoEncoderInstructor.to(net, 'cpu')
        torch.cuda.empty_cache()

        if cache is not None:
            cache.append(None)
            cache.append(mean_loss)  # 存储平均损失

        return 0, mean_loss # 返回值：准确率为空，损失为 mean_loss

    def fit(self):
        """
        重载fit方法，用于AutoEncoder任务
        """
        net, device = self.net, self.device
        lossf = self.lossf()

        AutoEncoderInstructor.to(net, device)
        if self.loader_sampler != None:
            self.batch_size = None
            self.epoch = 1
            loader = DataLoader(self.train_dataset,batch_sampler=self.loader_sampler)
        else:
            loader = DataLoader(self.train_dataset,self.batch_size,shuffle= True,pin_memory = True)

        opt = self.optc(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        losses = []
        net.train()

        for self.epochi in range(1, self.epoch + 1):

            mean_loss = 0

            for batch_counter,batch in enumerate(loader):
                opt.zero_grad()
                self.ontrainhandle(net) if self.ontrainhandle else None

                output = net(batch[0].to(device))
                # 重构部分
                # print(f"Output shape: {output.shape}, Target shape: {batch[0].shape}")
                loss = lossf(output, batch[0].to(device))
                loss.backward()
                self.gradhandle(net) if self.gradhandle else None

                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.clip_grad)
                    
                opt.step()
                self.updatedhandle(net) if self.updatedhandle else None

                # print(batch_counter,loss.detach().item())
                mean_loss += loss.detach().item()

            losses.append(mean_loss / (len(loader) + 1e-8))
            # print(f"Epoch {self.epochi}, Loss: {mean_loss / (len(loader) + 1e-8)}")

            if self.test_datset is not None:
                AutoEncoderInstructor.TestModel(net,self.test_datset,self.batch_size,device)
                AutoEncoderInstructor.to(net,device)

        AutoEncoderInstructor.to(net, 'cpu')
        return net, losses

def net(net_name):
    if net_name == 'cnnautoencoder':
        return AutoEncoder()
    else:
        return AEL()
    
NET = net
TRAINER = AutoEncoderInstructor
