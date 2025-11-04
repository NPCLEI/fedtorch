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

## Quick start

This repository includes a ready-to-run MNIST example under `experiments/MNIST/VisualizeProcess`. The MNIST example provides a simple script to run federated experiments and several configuration knobs to control data heterogeneity, participation and optimizer settings.

Run the example (two common ways):

```bash
# run in foreground
./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/MNIST/run.py

# or run in background and capture logs
nohup ./python3.13t/bin/python3.13t -Xgil=0 ./FederatedX/experiments/MNIST/VisualizeProcess/run.py > vp.file 2>&1 &
```

> The examples use a custom Python binary in the repository (`python3.13t` / `python3.14`). These custom builds are used to unlock improved threading behavior — adapt the path to the Python executable available in your environment if needed.

## Parameter tuning — two ways

You can tune experiment parameters in two ways: (1) edit the `CONFIG` dictionary in `experiments/<TASK>/config.py`, or (2) use command-line overrides at runtime.

**1) Edit `config.py` (recommended for reproducibility)**  
Modify the values under `client_trainer_arguments`, `server_arguments`, `cuda_argments`, etc., then save the file and run the experiment. This method is better for long-term tracking and sharing experiments because the configuration lives in a versioned file.

**2) Command-line overrides via `commands` list (used in `run.py`)**  
In this project we also support supplying runtime overrides through the `commands` list constructed in `run.py`. Each entry in `commands` is a tuple of `(ModuleOrTask, arg_fragment)` where `arg_fragment` is a formatted string describing which config section and key to override and the value to use. The experiment runner (`exp.run`) will expand these fragments, concatenate them, and call the target module with the assembled CLI — causing the specified `CONFIG` entries to be temporarily overridden for that run.

Example (from `experiments/MNIST/run.py`):

```python
ps = [0.01, 0.05, 0.1]
commands = [
    (VisPrc, f'.p -server_arguments --participation_rate_or_num .{p}') for p in ps
]

