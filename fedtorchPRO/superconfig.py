CONFIG = {
    "client_trainer_arguments":{
        "epoch" : 1,
        "lr" : 1e-2,
        "batch_size":32,
        "early_stop":-1,
        # "opt":"adam"
    },
    "server_arguments":{
        "commu_times" : 10,
        "participation_rate_or_num" : 0.01,
        "test_per_rounds":5,
        "loss_per_rounds":1,
    },
    "data":{
        "data_pickle_name":"emnist_cr.cn.200.alpha.0.1.npy",
        # "dirichlet_alpha" : -1,
        # "client_num" : 500,
        # "cur_ex_cls_num":100,
    },
    "cuda_argments":{
        "devices":{
            "cuda:0/0":-1,
            "cuda:0/1":-1,
            "cuda:0/2":-1,
            "cuda:0/3":-1,
        }
    },
    "FedProx":{
        "mu":0.01
    },
    "SCAFFOLD":{
        "slr":1e-1,
        "nl" :1e-1
    },
    "fedda":{
        "beta1":0.9,
        "beta2":0.99,
        "method" :"adam",
        "alpha":1,
        "slr"   :{
            "adam":1e-2,
            "sgdm":1e-2,
            "adagrad":1e-2,
        },
    },
    "ssgd":{
        "client_grd_num":20,
        "sample_weight_method":"dist", # loss,dist,
        "client_grd_num":20,
        "spn":5
    },
    "fedlin":{
        "topk":100
    },
    "fedopt":{
        "method":"adam",
        "server_lr":{
            "adam":1e-2,
            "adamw":1e-2,
            "adagrad":1e-2,
            "yogi":1e-2,
            "lamb":1e-2,
            "sgdm":1e-2,
            "sgd":1e-2,
        },
        "beta1":0.90,
        "beta2":0.99,
        "eps"  :1e-3,
        "eta_step" : 10,
        "error_feedback" : {
            "correct":True,
            "method":"scale",
            "topk":1/512,
            "error_decay":1,
            "rollback":False
        },
        "noise":False,
        "gamma":0.6
    },
    "fedwin":{
        "method":"adam",
        "nesterov" :False,
        "momentums":False,
        "reckless_steps":(2.0, 8.0),
        "server_lr":{
            "adam":1e-1,
            "adan":1e-2,
            "lamb":1e-2,
            "biprox":1e-2
        }
    },
    "fedmpx":{
        "method":"adam"
    },
    "fedprox":{
        "mu":0.01
    }
}

def save_(obj,name,root):
            
    import pickle
    f_name = "%s/%s.pickle"%(root,name)
    with open(f_name, 'wb+') as net_file:
        pickle.dump(obj,net_file)

class Parser:

    def __init__(self,config) -> None:
        self.config = config

    @staticmethod
    def eval(value):
        try:
            return eval(value)
        except:
            return value

    def _handle(self,sr:str):
        if '.' == sr[0]:
            self.exname = self.exname + sr
            sr = sr[1:]
        return sr

    def __call__(self,cmd:str):
        cmdsplited = cmd.split()
        self.exname = ''
        field = None

        while len(cmdsplited) > 0:
            curcmd = cmdsplited.pop(0)
            if curcmd[0] == '.':
                self._handle(curcmd)
            elif curcmd[0] == '-':
                if curcmd[1] == '-':
                    key = self._handle(curcmd[2:])
                    item = self._handle(cmdsplited.pop(0))
                    field[key] = Parser.eval(item)
                else:
                    field = SuperConfig.search_argment(self.config,self._handle(curcmd[1:]))

        self.exname = self.exname.removeprefix('.')
        return self.exname

# import mlflow
import importlib,os,pickle,torch
import random
from fedtorchPRO.utils.npcprint import npclog
from fedtorchPRO.core.cache import GPUManager,ModelManager

class SuperConfig:

    def __init__(self,config,command = None,run_id = None,run_name = 'lasted'):
        import sys,copy
        sys.path.append('./')
        self.command = command
        config = copy.deepcopy(config)
        self.parser = Parser(config)
        self.methodname = self.parser(command) if command else ''
        self.config = self.parser.config
        self.gpu_manager = GPUManager(self.devices)
        self.mdl_manager = ModelManager(self.NET)
        self.server = None 
        if run_id:
            self.run_id = run_id
            self.run_name = run_name

        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--debug', type=str, help='Enable debug mode')
        args = parser.parse_args()
        self.debug = args.debug == 'True'
        
    def print_config(self,config = None, indent=0):
        if config == None:
            config = self.config
        for key, value in config.items():
            print(' ' * indent + str(key) + ': ', end='')
            if isinstance(value, dict):
                print()  # 新行用于字典
                self.print_config(value, indent + 4)  # 递归调用，增加缩进
            else:
                print(value)

    def get(self,key,handle = None):
        res = SuperConfig.search_argment(self.config,key,True)
        if res == "no exits" and handle != None:
            return handle
        return res
    
    def weighted(self,close = True):
        self.config['workconfig']['close_weight_by_samle_nums'] = close

    def copynet(self,source,target = None):
        if target == None:
            target = self.NET()
        for pm,om in zip(target.parameters(),source.parameters()):
            pm.copy_(om)
        return target

    def flip_a_coin(self):
        if random.random() < self.get('coin_p',0.25):
            return 1
        else:
            return 0

    @staticmethod
    def search_argment(root:dict,res,ignore_no_exits =  False):
        stack = [(root,node) for node in root.keys()]
        while len(stack):
            root,node = stack.pop(0)
            if node == res:
                return root[node]
            elif type(root[node]) is dict:
                stack.extend([(root[node],child_node) for child_node in root[node].keys()])
        if ignore_no_exits:
            return "no exits"
        else:
            raise Exception(res,"no exits")

    def __getattr__(self, att_name):
        return self[att_name]

    def __getitem__(self,key):
        return SuperConfig.search_argment(self.config,key)
    
    def prepare_datas(arguments):
        import os
        from fedtorchPRO.data.utils import DataShell

        map_path = os.path.join(arguments['root_path'],arguments['exname'],arguments.datamapname)
        with open(map_path,'r',encoding='utf8') as fp:
            maps = eval(fp.read())

        DataShell.source = importlib.import_module(arguments['libs'] + 'dataset').get_source()
        return [DataShell(item) for item in maps]

    @property
    def available_gpus(self):
        return SuperConfig.search_argment(self.config,'devices')

    @property
    def pknum(self):
        pn = SuperConfig.search_argment(self.config,'participation_rate_or_num')
        if pn >= 1:
            return pn
        else:
            n = int(pn * SuperConfig.search_argment(self.config,'clients_num'))
            return max(n,1)
        
    @property
    def cudanum(self):
        return self.processnum

    @property    
    def testdata(self):
        return importlib.import_module(self['libs']+'dataset').get_test()

    @property    
    def traindata(self):
        return importlib.import_module(self['libs']+'dataset').get_source()

    @property
    def communic_times(self):
        return SuperConfig.search_argment(self.config,'commu_times')

    @property
    def processnum(self):
        cuda_argments = SuperConfig.search_argment(self.config,'cuda_argments')
        return len(cuda_argments['devices'])

    def inimodel(args):
        """
            Get ZERO Model
        """
        import os,torch
        work_path = os.path.join(args.root_path , args.exname)
        net_path = os.path.join(work_path,args.net_name)
        os.makedirs(work_path,exist_ok=True)
        net = args.NET()
        if os.path.exists(net_path):
            try:
                net.load_state_dict(torch.load(net_path,weights_only=True))
            except:
                torch.save(net.state_dict(), net_path)
        else:
            torch.save(net.state_dict(), net_path)
        return net
    
    @property
    def TRAINER(self):
        return importlib.import_module(self['libs']+'nn').TRAINER
    
    def NET(self) -> torch.nn.Module:
        """
        random init model
        """
        netf = importlib.import_module(self['libs']+'nn').NET
        return netf(self.net_name)

    @torch.no_grad()
    def ZeroNET(self,require_grad = False) -> torch.nn.Module:
        netf = importlib.import_module(self['libs']+'nn').NET
        net = netf(self.net_name)
        for pm in net.parameters(True):
            pm.fill_(0.0)
            pm.requires_grad_(require_grad)
        return net

    @property
    def TestModel(self):
        return importlib.import_module(self['libs']+'nn').TRAINER.TestModel
    
    @property
    def datamapname(self):
        name = f"clientsn.{self.clients_num}.alpha.{self.workconfig['alpha']}"
        return f"datas.map.{name}.txt"
    
    @property
    def workpath(self):
        return os.path.join(self.root_path,self.exname)

    @property
    def keepnet(self):
        return SuperConfig.search_argment(self.config,'keepnet',ignore_no_exits= True) == True

    def check_file(self,file_name,no_exist_handle = None,ignore_exception = False,root_path = None,delete = False):
        root_path = self['root_path'] + self['exname']
        file_name = file_name.replace('.pickle','')
        file_path = ("%s/%s"%(root_path,file_name)) + ('.pickle' if '.npy' not in file_name else '')
        if os.path.exists(file_path) and not delete:
            if 'pickle' in file_name:
                with open(file_path,"rb+") as model:
                    model = pickle.load(model)
                    npclog("file:<",file_path,"> exists... loaded")
                    return model
            else:
                from numpy import load
                return load(file_path,allow_pickle=True)
        elif no_exist_handle:
            npclog("file:",file_path,"does not exist.return user handler.")
            file = no_exist_handle(file_path)
            save_(file,file_name,root_path)
            return file
        elif ignore_exception:
            npclog("file:",file_path,"does not exist.return <NONE>.")
            return None
        else:
            raise Exception("file:",file_path)
    
    @property
    def run_path(self):       
        path = os.path.join(
            self.root_path,self.exname,
            f'A{str(self.alpha).replace('.','d')}C{self.clients_num}',self.run_name
        )
        os.makedirs(path,exist_ok=True)
        return path
    

    
    @property
    def baselines_path(self):       
        path = os.path.join(
            self.root_path,self.exname,
            f'A{str(self.alpha).replace('.','d')}C{self.clients_num}','baselines'
        )
        os.makedirs(path,exist_ok=True)
        return path

    def acquire_id(self,cid):
        return self.server.selected_ids.index(cid)

    @torch.no_grad()
    def acquire(self,cid):
        did = self.gpu_manager.acquire(cid)
        model = self.mdl_manager.acquire(self.server.selected_ids.index(cid))
        for target,source in zip(model.parameters(),self.server.global_model.parameters()):
            target.copy_(source)
        return did,model
