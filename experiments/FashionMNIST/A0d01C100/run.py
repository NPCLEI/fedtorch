import sys
sys.path.append('./')

if __name__ == '__main__':

    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/FashionMNIST/A0d01C100/run.py > ./mlruns/FashionMNIST.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/FashionMNIST/A0d01C100/run.py

    from fedtorchPRO import *

    commands = [
        # (FedAVGPlus,'.FedAVG .MPX'),
        # (FedAVGPlus,'.FedAVG .NAO'),
        # (FedAVGPlus,'.FedAVG'),
        # (DeltaSGD,'.DeltaSGD'),
        (FedEXP,'.FedExP'),
        (FedPROX,'.Prox'),
        ((FedAVG,DualProxMClient),'.DualProx'),
        (SCAFFOLD,'.SCAFFOLD'),
        # (FedSGDM,'.FedSGDM'),
        ((FedSGD,FedNAR),'.FedNAR'),
        (FedGM,'.FedGM'),
        (FedInit,'.FedInit'),
        (FedLESAM,'.FedLESAM'),
        (FedOPT,'.FedOPT -server_arguments --server_lr 1e-3'),
        (FedAVG,'.FedAVG'),
        # ((FedMPX,DualProxMClient),'.DualProx .MPX'),
        # (FedOPT,'.FedOPT .weighted -workconfig --close_weight_by_samle_nums True'),
        # (FedOPT,'.FedOPT .notweighted -workconfig --close_weight_by_samle_nums False'),
    ]

    exp.run(commands,True)


