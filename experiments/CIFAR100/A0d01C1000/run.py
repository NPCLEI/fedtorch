import sys
sys.path.append('./')

if __name__ == '__main__':

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR100/A0d01C1000/run.py > ./mlruns/cifar100log.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR100/A0d01C1000/run.py

    from fedtorchPRO import *

    commands = [
        (OursADP,f'.OursADP -hyerparams --.ours_grad_decay .0.99 -server_arguments --server_lr .1e-4'),
        (FedAVGPlus,'.FedAVG'),
        (FedADAM,'.FedADAM'),
        (FedSGDM,'.FedSGDM'),
        (FedAMS,'.FedADAM -fedopt --.ams True'),
        (FedEXP,'.FedEXP'),
        (FedGM,'.FedGM'),
        (FedLESAM,'.FedLESAM'),
    ]

    exp.run(commands,True)