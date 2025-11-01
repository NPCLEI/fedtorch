import sys
sys.path.append('./')

if __name__ == '__main__':

    from fedtorchPRO import *

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR100/GHDecetor/run.py > lastlog.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR100/GHDecetor/run.py
    # pkill -f gunicorn

    ps = [10] # + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [-1]

    commands = [
        (GHDetector,f'.p -server_arguments --participation_rate_or_num .{p}') for p in ps
    ]

    alphas = [0.001,0.01,0.1,1.0] # 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0] + [10]

    for alpha in alphas:
        exp.run(commands,alpha=alpha,clientn=100)