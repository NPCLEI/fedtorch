import sys
sys.path.append('./')

if __name__ == '__main__':

    from fedtorchPRO import *

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/A1d00C62/run.py > lastlog.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/A1d00C62/run.py
    # pkill -f gunicorn

    commands = [
        (GradiAnalyseServer,'.GradiAnalyseServer'),
    ]
    
    exp.run(commands)