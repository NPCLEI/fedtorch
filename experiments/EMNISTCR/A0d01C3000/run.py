import sys
sys.path.append('./')

if __name__ == '__main__':

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/A0d01C3000/run.py > ./mlruns/EMNISTCRA0d01C3000.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/A0d01C3000/run.py
    from fedtorchPRO import *
    from experiments.EMNISTCR.libs.nn import netmap

    commands = [
        (FedADAM,f".FedAdam -workconfig --net_name .{net_name}") for net_name in netmap.keys()
    ] + [ (FedADAM,f".FedAdam .MPX -workconfig --net_name .{net_name}") for net_name in netmap.keys() ]
    exp.run(commands,False)