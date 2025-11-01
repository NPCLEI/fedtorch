import sys
sys.path.append('./')

if __name__ == '__main__':

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/A0d01C100/run.py > ./mlruns/A0d01C100.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/A0d01C100/run.py

    from fedtorchPRO import *

    # T = '-fedopt --method adam -server_arguments --server_lr 1e-3'
    commands = [
        (Ours,'.Ours')
        # ((FedAVGPlus,DualProxMClient),f'.DualProx {T}'),
        # ((FedAVGPlus,DualProxMClient),f'.DualProx {T} .MPX'),
        # (FedAVGM,f'.FedAVG-M {T}'),
        # (FedAVGPlus,f'.FedAVG {T}'),
        # (DeltaSGD,f'.DeltaSGD {T}'),
        # (FedEXP,f'.FedExP {T}'),
        # (FedGM,f'.FedGM {T}'),
        # (FedInit,f'.FedInit {T}'),
        # (FedLESAM,f'.FedLESAM {T}'),
        # (FedNAR,f'.FedNAR {T}'),
        # (FedPROX,f'.DANE {T}'),
        # (SCAFFOLD,f'.SCAFFOLD {T}'),
        # (SCAFFNEW,f'.SCAFFNEW {T}'),
        # (FedADAM,f'.FedADAM {T}'),
    ]

    exp.run(commands,True)