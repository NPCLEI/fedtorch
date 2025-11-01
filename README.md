# fedtorch
A federated learning optimization framework. The framework consists of the following methods: FedAVG, FedAVGM, DeltaSGD, DualPROX, FedEXP, FedGM, FedInit, FedLESAM, FedNAR, FedPROX, SCAFFOLD, SCAFFNEW, and FedADAM

    The running examples are located in the experiments directory. Simply configure the appropriate parameters in the libs file and run experiments/MNIST/run.py.

This framework's key feature is its ability to utilize multiple GPUs while Python 3.14 unlocks thread locks, and to virtualize a single GPU into multiple GPUs for use! This means the framework taps into Python's true multithreading potential, significantly boosting training speed.

For example : experiments/\<specific task>/\<A: dirichlet alpha C: Clinets Num>/config.py:

    "cuda_argments":{
        "devices":[
            f"cuda:{i}/{j}" for i in range(4) for j in range(3)
        ]
    },

As long as it conforms to the path rules, the config file will be automatically read.