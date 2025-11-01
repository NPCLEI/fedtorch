CONFIG = {
    "workspace":{
        "root_path":  '/common/Workspace/',
        "exname":  'MNIST',
        "libs":"experiments.MNISTA.libs",
    },
    "filename":{
        "data_name":"emnist_cr_blc.datas.alpha.0.1.txt",
        "net_pickle_name" : "mlp"
     },
    "client_trainer_arguments":{
        "epoch" : 1,
        "lr" : 1e-3,
        "batch_size":32,
    },
    "server_arguments":{
        "commu_times" : 1000,
        "participation_rate_or_num" : 10,
        "test_batch_size":1024,
        "clients_num":3400,
        "server_lr":1e-3,
        "test_interval":30
    },
    "cuda_argments":{
        "devices":[
            f"cuda:0/{i}" for i in range(10)
        ]
    },
    "fedopt":{
        "method":"adam",
        "weight_decay":0.1,
        "momentum":0.9,
        "ams":False,
        "mpx":False
    }
}

def print_config(config, indent=0):
    for key, value in config.items():
        print(' ' * indent + str(key) + ': ', end='')
        if isinstance(value, dict):
            print()  # 新行用于字典
            print_config(value, indent + 4)  # 递归调用，增加缩进
        else:
            print(value)

# 使用示例
print_config(CONFIG)
