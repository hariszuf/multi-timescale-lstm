from pathlib import Path

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config


run_config = Config(Path("config.yml"))
print('model:\t\t', run_config.model)
print('use_frequencies:', run_config.use_frequencies)
print('seq_length:\t', run_config.seq_length)

print('dynamic_inputs:')
# run_config.dynamic_inputs


# by default we assume that you have at least one CUDA-capable NVIDIA GPU
def _main():
    if torch.cuda.is_available():
        start_run(config_file=Path("1_basin.yml"))
    # fall back to CPU-only mode
    else:
        start_run(config_file=Path("1_basin.yml"), gpu=-1)


if __name__ == "__main__":
    _main()
