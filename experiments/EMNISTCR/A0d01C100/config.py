CONFIG = {
    "workconfig":{
        "root_path":  './workspace/',
        "exname":  'EMNISTCR',
        "libs":"experiments.EMNISTCR.libs.",
        "net_name" : "cnn",
    },
    "client_trainer_arguments":{
        "epoch" : 10,
        "lr" : 1e-4,
        "batch_size":4096,
        "opt":"adam"
    },
    "server_arguments":{
        "commu_times" : 100,
        "participation_rate_or_num" : 0.1,
        "test_batch_size":8192,
        "server_lr":1e-4,
        "test_interval":15,
    },
    "cuda_argments":{
        "devices":[
            # f"cuda:{i}/{j}" for i in range(4) for j in range(3)
            f"cpu/{i}" for i in range(10)
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
        'fedmpx':0.2,
        'fednao':0.9,
        'prox':0.1,
        'dualprox':0.1,
        "fedlesam":0.1,
        "fedgm":(0.9,0.9)
    }
}

if __name__ == "__main__":
    import copy
    print(CONFIG)
    deep_copy = copy.deepcopy(CONFIG)
    CONFIG["client_trainer_arguments"] = 10
    print(CONFIG["client_trainer_arguments"])
    print(deep_copy["client_trainer_arguments"])