import sys
sys.path.append('./')

if __name__ == '__main__':

    from fedtorchPRO import *
    from fedtorchPRO.nn.instructor import EasyInstructor
    # nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/Cen/cen.py > lastlog.file 2>&1 &
    # ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/EMNISTCR/Cen/cen.py
    # pkill -f gunicorn

    scfg = exp.auto_load_config()
    net,_ = EasyInstructor(
        scfg.inimodel(),
        train_dataset = scfg.traindata,
        device='cuda:0',
        **scfg.client_trainer_arguments
    ).fit()
    scfg.client_trainer_arguments['lr']  = scfg.client_trainer_arguments['lr']/10
    scfg.client_trainer_arguments['opt'] = 'sgd'
    net,_ = EasyInstructor(
        net,
        train_dataset = scfg.traindata,
        device='cuda:0',
        **scfg.client_trainer_arguments
    ).fit()
    import torch
    torch.save(net.state_dict(),'./FederatedX/experiments/EMNISTCR/Cen/cen.state')