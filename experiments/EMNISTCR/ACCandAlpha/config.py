CONFIG = {
    "workconfig":{
        "root_path":  './workspace/',
        "exname":  'EMNISTCR',
        "libs":"experiments.EMNISTCR.libs.",
        "net_name" : "logical",
        # "keepnet":True
    },
    "client_trainer_arguments":{
        "epoch" : 1,
        "lr" : 1e-3,
        "batch_size":128,
        # "opt":"adam",
        "minibatch":True,
        # "util_loss":1e-1
    },
    "server_arguments":{
        "commu_times" : 1500,
        "participation_rate_or_num" : 0.01,
        "test_batch_size":5096,
        "server_lr":1e-3,
        "test_interval":10,
        "total_client_num":100
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