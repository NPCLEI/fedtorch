import sys
sys.path.append('./')

if __name__ == '__main__':

    from fedtorchPRO import *

    commands = [
        (FedAVG,'.FedAVG'),
    ]

    exp.run(commands)