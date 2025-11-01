import sys
sys.path.append('./')

if __name__ == '__main__':

    from fedtorchPRO.exp import exp
    from fedtorchPRO.core.server import Server
    from fedtorchPRO.ultra.opt.server import FedOPT
    from fedtorchPRO.ultra.prox.client import ProxClient
    from fedtorchPRO.ultra.dualproxm.client import DualProxMClient

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR10/A0d10C500/run.py > ./mlruns/lastlog.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR10/A0d10C500/run.py

    commands = [
        (FedOPT,'.FedOPT -fedopt --method .adam'),
        (FedOPT,'.FedOPT -fedopt --.ams True'),
        (FedOPT,'.FedOPT -fedopt --.nao True'),
        (FedOPT,'.FedOPT -fedopt --.mpx True'),
        ((Server,DualProxMClient),'.DualProx'),
        (Server,'.FedAVG'),
    ]

    exp.run(commands)