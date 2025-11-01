import sys
sys.path.append('./')

if __name__ == '__main__':

    from fedtorchPRO import *

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/A0d10C3000/run.py > lastlog.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/A0d10C3000/run.py
    # pkill -f gunicorn

    commands = [
        ((FedInit,FedInitClient),'.FedInit'),
        (FedAVG,'.FedAVG'),
        (FedEXP,'.FedExp'),
        # (FedOPTS,'.FedOPT'),
        # (FedOPTS,'.FedOPT -fedopt --.mpx True'),
        # (FedOPTS,'.FedOPT -fedopt --.ams True'),
        # (FedOPTS,'.FedOPT -fedopt --.nao True'),
        # ((Server,DualProxMClient),'.DualProx'),
    ]

    exp.run(commands)