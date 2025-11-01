
import torch

from tqdm.auto import tqdm
from sklearn import metrics

from torch.utils.data import DataLoader
from .optimizers import Adsgd

def set_seed(wid = 1):
    seed = 42
    import random
    import os
    import numpy as np
    # import transformers
    # transformers.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    try:
        from accelerate.utils import set_seed as auset_seed
        auset_seed(seed)
    except:
        pass

set_seed()

class EasyInstructor:

    def __init__(
            self,
            net,
            train_dataset,
            device,
            epoch = 1,
            lr = 1e-4,
            batch_size = 128,
            test_dataset = None,
            log_title = '',
            test_per_epoch = -1,
            lossf = 'ce',
            num_works = 1,
            opt = 'sgd',
            noise = False,
            early_stop = -1,
            loader_sampler = None,
            clip_grad = False,
            momentum = 0,
            weight_decay = 0,
            minibatch = False
        ):
        self.net = net
        self.train_dataset = train_dataset
        self.test_datset = test_dataset 
        self.device = device
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.log_title = log_title
        self.test_per_epoch = test_per_epoch
        self.print_train_acc = False
        self.num_works = num_works
        self.selectlossf(lossf)
        self.selectopt(opt)
        self.noise = noise
        self.loader_sampler = loader_sampler
        self.gradhandle = None
        self.updatedhandle = None
        self.ontrainhandle = None
        self.clip_grad = clip_grad
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.minibatch = minibatch

    def selectopt(self,opt):
        if type(opt) is str:
            if 'sgd' == opt.lower():
                self.optc = torch.optim.SGD
            elif 'adamw' == opt.lower():
                self.optc = torch.optim.AdamW
            elif 'adam' == opt.lower():
                self.optc = torch.optim.Adam
            elif 'adsgd' == opt.lower():
                self.optc = Adsgd
            elif 'lamb' == opt.lower():
                from torch_optimizer import Lamb
                self.optc = Lamb
        else:
            self.optc = opt

    def selectlossf(self,lossf):
        if 'mse' in lossf:
            self.lossf = torch.nn.MSELoss
        elif 'bce' in lossf:
            self.lossf = torch.nn.BCELoss
        else:
            self.lossf = torch.nn.CrossEntropyLoss

    @staticmethod
    def TestModel(net,test_dataset,batch_size = 2048,device = 'cuda:0',commic_times = -1):
        return EasyInstructor._test_model_(net,test_dataset,batch_size,device,commic_times)

    @staticmethod
    @torch.no_grad()
    def _test_model_(net,test_dataset,batch_size = 64,device = 'cuda:0',commic_times = -1,cache = None,num_workers = 4):
        set_seed()

        if test_dataset == None:
        
            if cache != None:
                cache.append(-1)
                cache.append(None)

            return -1,None
        
        targets = []
        outputs = []
        EasyInstructor.to(net,device)
        loader = DataLoader(test_dataset,batch_size,shuffle=False,num_workers=num_workers) # num_workers=2
        for batch in tqdm(loader):

            o = net(batch[0].to(device))

            targets.extend(batch[1].tolist())
            outputs.extend(o.detach().argmax(-1).cpu().tolist())

        acuv = metrics.accuracy_score(targets, outputs)
        metx = metrics.classification_report(targets, outputs,zero_division=1)
        
        print(metx)
        print(acuv)

        EasyInstructor.to(net,'cpu')
        
        torch.cuda.empty_cache()
        if cache != None:
            cache.append(acuv)
            cache.append(metx)
        
        return acuv,metx
        
    @staticmethod
    def lossfn(net,batch,device,lossf = None):
        """
        core compute
        Args:
            net:net,
            batch:
        Return:
            loss:
            output:
        """
        if lossf == None:
            lossf = torch.nn.functional.cross_entropy
        output = net(batch[0].to(device))
        loss = lossf(output,batch[1].long().to(device))
        return loss,output

    def log_metic(self,mean_loss,right = 0):
        print(
                '[%s] epoch:%d'%(self.log_title,self.epochi),
                ' train acc:%2.3f'%100*right/len(self.train_dataset) if self.print_train_acc else '',
                ' train loss:',mean_loss,'\n',"-"*30
            )

    @staticmethod
    def to(md,device):

        if md == None:
            return md

        if type(device) == str:
            device = torch.device(device)
        elif issubclass(type(device),torch.nn.Module):
            for pm in device.parameters():
                device = pm.device
                break
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

    def fit(self):
        
        net,device = self.net,self.device
        lossf = self.lossf()

        EasyInstructor.to(net,device)
        if self.loader_sampler != None:
            self.batch_size = None
            self.epoch = 1
            loader = DataLoader(self.train_dataset,batch_sampler=self.loader_sampler)
        else:
            loader = DataLoader(self.train_dataset,self.batch_size,shuffle= True,pin_memory = True) 

        opt = self.optc(net.parameters(), lr=self.lr,weight_decay=self.weight_decay) # ,momentum=0.9, weight_decay=5e-4

        losses = []
        net.train()
        minibatch_k = 0

        for self.epochi in range(1,self.epoch +1 ):

            mean_loss = 0
            right = 0

            for batch_counter,batch in enumerate(loader):
                opt.zero_grad()
                self.ontrainhandle(net) if self.ontrainhandle else None
                output = net(batch[0].to(device))
                loss = lossf(output,batch[1].to(device))
                loss.backward()
                self.gradhandle(net) if self.gradhandle else None
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.clip_grad)
                opt.step()
                self.updatedhandle(net) if self.updatedhandle else None
                if self.print_train_acc:
                    right += (output.argmax(-1).detach().cpu() == batch[1]).sum().item()
                    print(self.epochi,100 * batch_counter/len(loader),'loss:',loss.detach().item())

                mean_loss += loss.detach().item()

                if self.minibatch:
                    minibatch_k += 1
                
                if minibatch_k > self.epoch:
                    break

            losses.append(mean_loss/(len(loader)+1e-8))

            if minibatch_k > self.epoch:
                break

            # self.log_metic(mean_loss/(len(loader)+1e-8),right)

            if self.test_datset != None:
                EasyInstructor.TestModel(net,self.test_datset,self.batch_size,device)
                EasyInstructor.to(net,device)

        EasyInstructor.to(net,'cpu')

        return net,losses