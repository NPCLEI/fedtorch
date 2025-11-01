CONFIG = {
    "workconfig":{
        "root_path":  './workspace/',
        "exname":  'CIFAR100',
        "libs":"experiments.CIFAR100.libs.",
        "net_name" : "vitsmall",
    },
    "client_trainer_arguments":{
        "epoch" : 3,
        "lr" : 1e-3,
        "batch_size":196,
    },
    "server_arguments":{
        "commu_times" : 3000,
        "participation_rate_or_num" : 0.01,
        "test_batch_size":1024,
        "server_lr":1e-4,
        "test_interval":30,
    },
    "cuda_argments":{
        "devices":[
            f"cuda:{i}/{j}" for i in range(4) for j in range(4)
        ]
    },
    "fedopt":{
        "method":"adam",
        "weight_decay":0.1,
        "momentum":0.9,
        "ams":False,
        "mpx":False,
        "nao":False,
        "betas":(0.9,0.999)
    },
    "hyerparams":{
        'fedmpx':0.5,
        'fednao':0.9,
        'prox':0.01,
        'fednar':0.1,
        'dualprox':0.1,
        "fedlesam":0.01,
        "fedgm":(0.9,0.9),
        "ours_grad_decay":0.1
    }
}

if __name__ == "__main__":
    import copy
    print(CONFIG)
    deep_copy = copy.deepcopy(CONFIG)
    CONFIG["client_trainer_arguments"] = 10
    print(CONFIG["client_trainer_arguments"])
    print(deep_copy["client_trainer_arguments"])