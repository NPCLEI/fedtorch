import sys
sys.path.append('./')

if __name__ == '__main__':

    from fedtorchPRO import *

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR10/A0d10C300/run.py > lastlog.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR10/A0d10C300/run.py

    commands = [
        # (Server,'.FedAVG'),
        (FedOPT,'.FedOPT'),
        ((Server,DualProxMClient),'.DualProx'),
        (FedOPT,'.FedOPT -fedopt --.mpx True'),
        (FedOPT,'.FedOPT -fedopt --.ams True'),
        (FedOPT,'.FedOPT -fedopt --.nao True'),
    ]

    exp.run(commands)