CONFIG = {
    "workconfig":{
        "root_path":  './workspace/',
        "exname":  'FashionMNIST',
        "libs":"experiments.FashionMNIST.libs.",
        "net_name" : "cnn",
    },
    "client_trainer_arguments":{
        "epoch" : 5,
        "lr" : 1e-3,
        "batch_size":512,
    },
    "server_arguments":{
        "commu_times" : 3000,
        "participation_rate_or_num" : 0.1,
        "test_batch_size":5096,
        "server_lr":1e-1,
        "test_interval":10,
    },
    "cuda_argments":{
        "devices":[
            # f"cpu/{i}" for i in range(30)
            f"cuda:{i}/{j}" for i in range(4) for j in range(2)
        ]
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
        'fedmpx':1.0,
        'fednao':0.9,
        'prox':0.01,
        'fednar':0.01,
        'dualprox':0.01,
        'fedlesam':0.1,
        'fedgm':(0.1,0.1)
    }
}

if __name__ == "__main__":
    import copy
    print(CONFIG)
    deep_copy = copy.deepcopy(CONFIG)
    CONFIG["client_trainer_arguments"] = 10
    print(CONFIG["client_trainer_arguments"])
    print(deep_copy["client_trainer_arguments"])