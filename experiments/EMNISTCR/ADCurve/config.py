CONFIG = {
    "workconfig":{
        "root_path":  './workspace/',
        "exname":  'EMNISTCR',
        "libs":"experiments.EMNISTCR.libs.",
        "net_name" : "cnn",
        "close_weight_by_samle_nums":True,
        "x_start":''
    },
    "client_trainer_arguments":{
        "epoch" : 3,
        "lr" : 1e-3,
        "batch_size":1024,
    },
    "server_arguments":{
        "commu_times" : 1,
        "participation_rate_or_num" : -1,
        "test_batch_size":5096,
        "server_lr":1e-1,
        "test_interval":10,
    },
    "cuda_argments":{
        #  "devices":['cpu/0']
         "devices":[f"cuda:{i}/{j}" for i in range(3) for j in range(20)],
        # "devices":[
        #     f"cpu/{i}" for i in range(60)
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