import sys
sys.path.append('./')

if __name__ == '__main__':

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/A0d01C10000/run.py > lastlog2.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/A0d01C10000/run.py

    from fedtorchPRO import *

    commands = [
        # ((Server,DualProxMClient),'.DualProx'),
        (Server,'.FedAVG'),
        (FedEXP,'.FedExp'),
        # (FedOPTS,'.FedOPT'),
        # (FedOPTS,'.FedOPT -fedopt --.ams True'),
        # (FedOPTS,'.FedOPT -fedopt --.nao True'),
    ]

    exp.run(commands)