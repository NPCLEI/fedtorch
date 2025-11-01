import sys
sys.path.append('./')

if __name__ == '__main__':

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR10/A0d10C100/run.py > lastlog.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR10/A0d10C100/run.py

    from fedtorchPRO import *
    
    commands = [
        (Server,'.FedAVG'),
        # (FedOPT,'.FedOPT -fedopt --method .lamb'),
        # (FedOPT,'.FedOPT -fedopt --.mpx True'),
        # (FedOPT,'.FedOPT -fedopt --.ams True'),
        # (FedOPT,'.FedOPT -fedopt --.nao True'),
        # ((Server,DualProxMClient),'.DualProx'),
    ]

    exp.run(commands)