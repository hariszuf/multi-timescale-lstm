import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run


def _main():
    # by, default we assume that you have at least one CUDA-capable NVIDIA GPU
    if torch.cuda.is_available():
        start_run(config_file=Path("1_basin.yml"))

    # fall back to CPU-only mode
    else:
        start_run(config_file=Path("1_basin.yml"), gpu=-1)


if __name__ == "__main__":
    _main()
