CONFIG = {
    "workconfig":{
        "root_path":  './workspace/',
        "exname":  'CIFAR100',
        "libs":"experiments.CIFAR100.libs.",
        "net_name" : "vitsmall",
    },
    "client_trainer_arguments":{
        "epoch" : 10,
        "lr" : 1e-4,
        "batch_size":128,
        "opt":"adam",
        "minibatch":True
        # "util_loss":1e-1
    },
    "server_arguments":{
        "commu_times" :  3000,
        "participation_rate_or_num" : -1,
        "test_batch_size":512,
        "server_lr":1e-3,
        "test_interval":30,
    },
    "cuda_argments":{
        #  "devices":['cpu/0']
        "devices":[f"cuda:{i}/{j}" for i in range(4) for j in range(1)],
        # "devices":[
        #     f"cpu/{i}" for i in range(10)
        # ]
    },
    "fedopt":{
        "method":"adam",
        "weight_decay":0.1,
        "momentum":0.9,
        "ams":False,
        "mpx":False,
        "nao":False,
    },
    "hyerparams":{
        'fedmpx':1,
        'fednao':0.9,
        'prox':0.1,
        'dualprox':0.1,
        "fedinit" : 0.1
    }
}

if __name__ == "__main__":
    import copy
    print(CONFIG)
    deep_copy = copy.deepcopy(CONFIG)
    CONFIG["client_trainer_arguments"] = 10
    print(CONFIG["client_trainer_arguments"])
    print(deep_copy["client_trainer_arguments"])