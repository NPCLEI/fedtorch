import sys
sys.path.append('./')

if __name__ == '__main__':

    from fedtorchPRO import *

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR10/A0d01C1000/run.py > lastlog.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR10/A0d01C1000/run.py

    from fedtorchPRO import *

    commands = [
        (OursADP,f'.OursADP -hyerparams --.ours_grad_decay .0.9 -server_arguments --server_lr .1e-3'),
        (FedAVGPlus,'.FedAVG'),
        (FedADAM,'.FedADAM'),
        (FedSGDM,'.FedSGDM'),
        (FedAMS,'.FedADAM -fedopt --.ams True'),
        (FedEXP,'.FedEXP'),
        (FedGM,'.FedGM'),
        (FedLESAM,'.FedLESAM'),
    ]
    exp.run(commands,True)