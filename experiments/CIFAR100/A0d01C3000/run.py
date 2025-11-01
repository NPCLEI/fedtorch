import sys
sys.path.append('./')

if __name__ == '__main__':

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR100/A0d01C3000/run.py > cifar100log.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/CIFAR100/A0d01C3000/run.py

    from fedtorchPRO import *
    T = '-fedopt --method adam'
    commands = [
        (Ours,f'.Ours -fedopt --betas (0.0,0.999) -hyerparams --ours_grad_decay .{0.9}'),
        (Ours,f'.Ours -fedopt --betas (0.0,0.999) -hyerparams --ours_grad_decay .{0.9} .MPX'),
        (FedADAM,'.FedADAM .MPX'),
        (FedAVGPlus,f'.FedAVG'),
        (FedOPT,f'.FedLAMB -fedopt --method lamb  -server_arguments --server_lr 1e-3'),
        (FedADAM,f'.FedADAM {T} .MPX -hyerparams --fedmpx .0.2'),
        (FedAVGM,f'.FedAVGM {T}'),
        (DeltaSGD,f'.DeltaSGD {T}'),
        (DualPROX,f'.DualPROX {T}'),
        (FedEXP,f'.FedEXP'),
        (FedGM,f'.FedGM'),
        (FedInit,f'.FedInit {T}'),
        (FedLESAM,f'.FedLESAM {T}'),
        (FedNAR,f'.FedNAR {T}'),
        (FedPROX,f'.DANE {T}'),
        (SCAFFOLD,f'.SCAFFOLD {T}'),
        (SCAFFNEW,f'.SCAFFNEW {T}'),
    ]
    exp.run(commands)