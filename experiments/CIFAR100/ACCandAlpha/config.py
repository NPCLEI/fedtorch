CONFIG = {
    "workconfig":{
        "root_path":  './workspace/',
        "exname":  'CIFAR100',
        "libs":"experiments.CIFAR100.libs.",
        "net_name" : "vitsmall",
        # "keepnet":True
    },
    "client_trainer_arguments":{
        "epoch" : 4,
        "lr" : 1e-3,
        "batch_size":32,
        # "opt":"adam",
        "minibatch":True,
        # "util_loss":1e-1
    },
    "server_arguments":{
        "commu_times" : 3000,
        "participation_rate_or_num" : 0.01,
        "test_batch_size":1024,
        "server_lr":1e-3,
        "test_interval":50,
    },
    "cuda_argments":{
        "devices":[f"cuda:{i}/{j}" for i in range(4) for j in range(10)]
        # "devices":[
        #     f"cpu/{i}" for i in range(62)
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
        'fedmpx':0.5,
        'fednao':0.9,
        'prox':0.1,
        'dualprox':0.1
    }
}

if __name__ == "__main__":
    import copy
    print(CONFIG)
    deep_copy = copy.deepcopy(CONFIG)
    CONFIG["client_trainer_arguments"] = 10
    print(CONFIG["client_trainer_arguments"])
    print(deep_copy["client_trainer_arguments"])