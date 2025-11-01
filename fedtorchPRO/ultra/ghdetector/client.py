import torch

from fedtorchPRO import Client
from fedtorchPRO import ModelTool

class GDHClient(Client):

    def fullbatchgradi(self,server):

        device_id,net,batch_size = server.acquire(self.cid)
        device = device_id.split('/')[0]
        from torch.utils.data import DataLoader
        lossf = torch.nn.CrossEntropyLoss()
        ModelTool.to(net,device)
        loader = DataLoader(self.train_dataset,batch_size) 
        opt = torch.optim.Adam(net.parameters(),1)
        opt.zero_grad()
        net.train()

        for batch in loader:
            output = net(batch[0].to(device))
            loss = lossf(output,batch[1].to(device))
            loss.backward()

        with torch.no_grad():
            for pm in net.parameters():
                pm.copy_(pm.grad)

        ModelTool.to(net,'cpu')

        self.gradi = net

        server.gpu_manager.release(device_id,self.cid)

        return None
    
    def TrainModel(self, arguments):
        return super().TrainModel(arguments)