CONFIG = {
    "workconfig":{
        "root_path":  './workspace/',
        "exname":  'EMNISTAE',
        "libs":"experiments.EMNISTAE.libs.",
        "net_name" : "cnnautoencoder",
    },
    "client_trainer_arguments":{
        "epoch" : 10,
        "lr" : 1e-1,
        "batch_size":512,
        "opt":"adam",
    },
    "server_arguments":{
        "commu_times" : 300,
        "participation_rate_or_num" : 10,
        "test_batch_size":4096,
        "server_lr":1e-2,
        "test_interval":10,
    },
    "cuda_argments":{
        "devices":[
            # f"cuda:{i}/{j}" for i in range(4) for j in range(10)
            f"cpu/{j}"for j in range(20)
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