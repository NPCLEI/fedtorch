import sys
sys.path.append('./')

if __name__ == '__main__':

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/A0d01C20/run.py > ./mlruns/A0d01C100.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/A0d01C20/run.py

    from fedtorchPRO import *

    commands = [
        (SCAFFNEW,".scaffnew")
    ]

    exp.run(commands,False)