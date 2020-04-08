# FusedProp

Read the paper [here](https://arxiv.org/abs/2004.03335).

## Installation

- Clone this repository
- Install pip packages:
    - torch
    - torchvision
    - tensorflow (for FID/IS calculation)
    - wandb
    - numpy
    - scipy
    - tqdm
    - imageio
- For Weights and Biases (wandb), follow install/setup instructions [here](https://docs.wandb.com/quickstart). If you want to disable wandb, you can set the env variable `WANDB_MODE=dryrun`, although there is no alternate logging.
- Download the FID statistics of the test set following the instructions [here](https://github.com/TAMU-VITA/AutoGAN#prepare-fid-statistic-file). Downlaod to `[DATA_ROOT]/fid_stat/fid_stats_[DATASET]_train.npz`, where `DATA_ROOT` defaults to `.`.

## Usage:

Train a model by running a command a command similar to the following from the root directory of this repository:
`python -m gr_gan.launch --dataset=cifar10 --model=resnet --loss=nonsaturating --lr=.0001 --lr_dis=.0004 --z_dim=128 --batch_size=64 --train_fn=baseline --evaluate_freq=5000 --iterations=100000 --log_freq=10 --spectral_norm=True`

Descriptions of all command-line options are available by running:
`python -m gr_gan.trainable --help`
