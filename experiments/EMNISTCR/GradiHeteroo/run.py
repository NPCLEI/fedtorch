import sys
sys.path.append('./')

if __name__ == '__main__':

    from fedtorchPRO import *

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/DoubleMerge/run.py > lastlog.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/GHDector/run.py
    # pkill -f gunicorn

    commands = [
        (GHDetector,''),
    ]

    for alpha in [i/100 + 0.01 for i in range(100)]:
        exp.run(commands,alpha=alpha,clientn=62)