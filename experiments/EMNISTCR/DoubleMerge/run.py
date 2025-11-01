import sys
sys.path.append('./')

if __name__ == '__main__':

    from fedtorchPRO import *

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/DoubleMerge/run.py > lastlog.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/DoubleMerge/run.py
    # pkill -f gunicorn

    commands = [
        (FedADAM,'.FedADAM'),
    ]

    for alpha in [0.1,1,10,0.001,0.01]:
        exp.run(commands,alpha=alpha,clientn=62)