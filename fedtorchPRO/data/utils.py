
import torch
import numpy as np

from torch.utils.data import Dataset as torch_dataset
from fedtorchPRO.superconfig import SuperConfig

class DataShell(torch_dataset):
    
    source = None

    def __init__(self,mapidx):
        super(DataShell, self).__init__()
        self.mapidx = mapidx
    
    def __len__(self):
        return len(self.mapidx)
    
    def __getitem__(self, idx):
        if idx >= len(self.mapidx):
            raise StopIteration()
        return DataShell.source[self.mapidx[idx]]
    
def _split_noniid_data(source,scfg:SuperConfig):
    """
        Params:
            expth : 数据保存文件夹，建议保存到实验目录下
            source: 源数据，待分配的数据
            clients_num:客户端数量
            alpha : 狄利克雷分布参数
    """
    
    clients_num = scfg.clients_num
    alpha = scfg.workconfig['alpha']
    expth = scfg.workpath

    num_classes = torch.unique(source.targets).shape[-1]
    # 统计每个类别的下标
    dmtx = torch.tensor(np.random.dirichlet(np.ones(clients_num)*alpha, num_classes)).T
    class_indices = [torch.where(source.targets == i)[0] for i in range(num_classes)]
    datas = [[] for i in range(clients_num)]
    cidss = [i for i in range(clients_num)]

    while sum([len(i) for i in class_indices]) > 0:
        for cid in cidss:
            client,clientp = [],dmtx[cid]
            datalens = (clientp * torch.tensor([len(i) for i in class_indices])).ceil()
            for i in range(len(datalens)):
                nums = datalens[i].int().item()
                dist = class_indices[i][:nums]
                client.extend(dist.tolist())
                class_indices[i] = class_indices[i][nums:]
            datas[cid].extend(client)
        np.random.shuffle(cidss)

    for client in datas:
        if len(client) <= 2:
            maxci = np.argmax([len(c) for c in datas])
            rnums = np.random.randint(2,max(int(len(datas[maxci]) * 0.1),3))
            client.extend(datas[maxci][:rnums])

    import os

    os.makedirs(expth,exist_ok=True)

    name = f"clientsn.{clients_num}.alpha.{alpha}"

    with open(os.path.join(expth,f"datas.map.{name}.txt"),'w+',encoding='utf8') as fp:
        fp.write(str(datas))

    print('save to',f"datas.map.{name}.txt")

    with open(os.path.join(expth,f"distribute.matirx.{name}.txt"),'w+',encoding='utf8') as fp:
        fp.write(str(dmtx))
    print('save to',f"distribute.matirx.{name}.txt")


def split_noniid_data(source,scfg:SuperConfig):
    """
        Params:
            expth : 数据保存文件夹，建议保存到实验目录下
            source: 源数据，待分配的数据
            clients_num:客户端数量
            alpha : 狄利克雷分布参数
    """
    
    clients_num = scfg.clients_num
    alpha = scfg.workconfig['alpha']
    expth = scfg.workpath
  
    targets = torch.tensor(source.targets,dtype=torch.int) 
    
    num_classes = torch.unique(targets).shape[-1]
    # 统计每个类别的下标
    dmtx = torch.tensor(np.random.dirichlet(np.ones(clients_num)*alpha, num_classes))
    class_indices = [torch.where(targets == i)[0] for i in range(num_classes)]
    datas = [[] for i in range(clients_num)]
    cidss = [i for i in range(clients_num)]

    while sum([len(i) for i in class_indices]) > 0:
        np.random.shuffle(cidss)
        for i,clsp in enumerate(dmtx):
            dnums = (clsp * class_indices[i].shape[-1]).ceil().int()
            for cid in cidss:
                datas[cid].extend(class_indices[i][:dnums[cid]].tolist())
                if dnums[cid] <= len(class_indices[i]):
                    class_indices[i] = class_indices[i][dnums[cid]:]

    for client in datas:
        if len(client) <= 2:
            maxci = np.argmax([len(c) for c in datas])
            rnums = np.random.randint(2,max(int(len(datas[maxci]) * 0.1),3))
            client.extend(datas[maxci][:rnums])

    import os

    os.makedirs(expth,exist_ok=True)

    name = f"clientsn.{clients_num}.alpha.{alpha}"

    with open(os.path.join(expth,f"datas.map.{name}.txt"),'w+',encoding='utf8') as fp:
        fp.write(str(datas))

    print('save to',f"datas.map.{name}.txt")

    with open(os.path.join(expth,f"distribute.matirx.{name}.txt"),'w+',encoding='utf8') as fp:
        fp.write(str(dmtx.tolist()))
    print('save to',f"distribute.matirx.{name}.txt")

    # exit()

def generate_root_data(source,root_dataset_size,root_dataset_bias):
    """

    """
    import random

    targets = torch.tensor(source.targets,dtype=torch.int) 
    
    num_classes = torch.unique(targets).shape[-1]
    # 统计每个类别的下标
    class_indices = [torch.where(targets == i)[0] for i in range(num_classes)]
    distributed = [0 for i in class_indices]
    class_num = len(class_indices)
    res = []

    for i in range(root_dataset_size):
        rand = random.random()
        if rand < root_dataset_bias:
            label = 0
        else:
            label = random.randint(0,class_num-1)
        res.append(class_indices[label][distributed[label]])
        distributed[label] = distributed[label] + 1

    return res
