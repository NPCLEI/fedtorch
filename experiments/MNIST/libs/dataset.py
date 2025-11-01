
import torch

from fedtorchPRO.imitate.datasets.mnist import EMNIST
from fedtorchPRO.imitate import transforms 

class Div255(torch.nn.Module):
    def forward(self,x:torch.Tensor):
        return x.div_(255.0)

def get_source():
    return EMNIST( 
        root='./datasets/EMNIST/',
        train=True,
        download=False,
        transform = transforms.Compose([
            transforms.ToTensor(),
            Div255()
        ]),
        split='digits' 
    )

def get_test():
    return EMNIST( 
        root='./datasets/EMNIST/',
        train=False,
        download=False,
        transform = transforms.Compose([
            transforms.ToTensor(),
            Div255()
        ]),
        split='digits' 
    )

# from fedtorchPRO.superconfig import SuperConfig

# def split_noniid_data(source,scfg:SuperConfig):
#     """
#         Params:
#             expth : 数据保存文件夹，建议保存到实验目录下
#             source: 源数据，待分配的数据
#             clients_num:客户端数量
#             alpha : 狄利克雷分布参数
#     """


#     from fedtorchPRO.data.utils import NPCDataset
#     from fedtorchPRO.utils.distribute import FedUtils,FedIDXDistributer
    
#     clients_num = scfg.clients_num
#     alpha = scfg.workconfig['alpha']
#     expth = scfg.workpath

#     train = NPCDataset(source,True)

#     dmtx = FedUtils.DistributeMatrix(clients_num=clients_num,label_num=train.cls_num,alpha=alpha)
#     while not FedIDXDistributer.suitable(dmtx,train):
#         dmtx = FedUtils.DistributeMatrix(clients_num=clients_num,label_num=train.cls_num,alpha=alpha)
#     distributer = FedIDXDistributer(train=train,dmtx=dmtx)
#     datas = distributer.distribute()

#     import os

#     os.makedirs(expth,exist_ok=True)

#     name = f"clientsn.{clients_num}.alpha.{alpha}"

#     with open(os.path.join(expth,f"datas.map.{name}.txt"),'w+',encoding='utf8') as fp:
#         fp.write(str(datas))

#     print('save to',f"datas.map.{name}.txt")

#     with open(os.path.join(expth,f"distribute.matirx.{name}.txt"),'w+',encoding='utf8') as fp:
#         fp.write(str(dmtx))
#     print('save to',f"distribute.matirx.{name}.txt")

#     exit()
