
import torch

from tqdm.auto import tqdm
from sklearn import metrics
from torch.utils.data import DataLoader,RandomSampler

from npcfedfmk.modulesx.nns import EasyInstructor
from npcfedfmk.panel import CTRL

class SCAFFOLDInstructor(EasyInstructor):

    def fit(self):
        
        net,sc,cc = self.net
        device = self.device
        EasyInstructor.to(net,device)
        EasyInstructor.to(sc,device)
        EasyInstructor.to(cc,device)

        if hasattr(self.train_dataset,'generate'):
            self.train_dataset.batch_size = self.batch_size
            loader = self.train_dataset
        else:
            loader = DataLoader(self.train_dataset,self.batch_size,shuffle=True) #,num_workers=2

        opt = torch.optim.SGD(net.parameters(), lr=self.lr)
        lossf = self.lossf()

        losses = []
        net.train()
        
        for self.epochi in range(1,self.epoch +1):

            mean_loss = 0
            right = 0
            for batch in loader:
                opt.zero_grad()
                
                loss,output = CTRL.trainer_class.lossfn(net,batch,device,lossf)
                
                loss.backward()

                for g,ci,c in zip(net.parameters(),cc.parameters(),sc.parameters()):
                    g.grad = g.grad - ci + c

                opt.step()
                
                mean_loss += loss.detach().item()
                if self.print_train_acc:
                    right += (output.argmax(-1).detach().cpu() == batch[1]).sum().item()

            losses.append(mean_loss/len(loader))
            self.log_metic(mean_loss/len(loader),right)

        EasyInstructor.to(net,'cpu')
        EasyInstructor.to(sc,'cpu')
        EasyInstructor.to(cc,'cpu')

        return net,losses

