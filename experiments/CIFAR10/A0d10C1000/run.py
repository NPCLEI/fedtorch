import sys
sys.path.append('./')

if __name__ == '__main__':

    from fedtorchPRO.exp import exp
    from fedtorchPRO.core.server import Server
    from fedtorchPRO.superconfig import SuperConfig
    from fedtorchPRO.ultra.opt.server import FedOPT
    from fedtorchPRO.ultra.prox.client import ProxClient
    from fedtorchPRO.ultra.dualproxm.client import DualProxMClient

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR10/A0d10C1000/run.py > lastlog.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR10/A0d10C1000/run.py

    commands = [
        (FedOPT,'.FedOPT'),
        (FedOPT,'.FedOPT -fedopt --.mpx True'),
        (Server,'.FedAVG'),
        (FedOPT,'.FedOPT -fedopt --.ams True'),
        (FedOPT,'.FedOPT -fedopt --.nao True'),
        ((Server,DualProxMClient),'.DualProx'),
    ]

    exp.run(commands)