# Code Setup

## Environment

Create environments by using the provided environment files.

`conda env create --file environment_<OS>.yml --name <NAME>`


After creating the environment, you can check the installed pytorch and cuda version with:

```
import torch
torch.__version__ 
torch.version.cuda 
```

If you need a different version for your setup, install it via: 

`pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 --index-url https://download.pytorch.org/whl/cu121`


If there are issues with the dependencies `torch_scatter` or `torch_cluster`, install them via a command similar to this one (adjust the torch and cuda version to your setup):

`pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.1.1+cu121.html`

# Configuration

Training models with this codebase requires some configurations such as where the code can find datasets or where to put logs.

## static_config.yaml

The file `static_config.yaml` defines all kinds of paths that are specific to your setup:
- where to store checkpoints/logs (`output_path`)
- from where to load data (`global_dataset_paths`)

Some additional configurations are contained:
- `local_dataset_path` if this is defined, data will be copied before training to this location. This is typically used 
if there is a "slow" global storage and compute nodes have a fast local storage (such as a fast SSD).
- `default_wandb_mode` how you want to log to [Weights and Biases](https://wandb.ai/)
  - `disabled` dont log to W&B
  - `online` use W&B in the "online" mode, i.e. such that you can see live updates in the web interface
  - `offline` use W&B in the "offline" mode. This has to be used if compute nodes dont have internet access. You can 
sync the W&B logs after the run has finished to inspect it via the web interface.
  - if `online` or `offline` is used you will need to create a `wandb_config` (see below)

To get started copy `template_static_config_github.yaml`, rename it to `static_config.yaml` and adapt it to your setup.

## W&B config

You can log to W&B by setting a `wandb_mode`. Set it in the `static_config.yaml` via `default_wandb_mode`. 
You can define to which W&B project you want to log to via a `wandb: <CONFIG_NAME>` field in a yaml file that defines your run.

All provided yamls by default use the name `v4` as `<CONFIG_NAME>`. To use the same config as defined in the provided 
yamls create a folder `wandb_configs`, copy the `template_wandb_config.yaml` into this folder, change 
`entity`/`project` in this file and rename it to `cvsim.yaml`.
Every run that defines `wandb: cvsim` will now fetch the details from this file and log your metrics to this W&B project.

## SLURM config

This codebase supports runs in SLURM environments. For this, you need to provide some additional configurations.
Copy the `template_sbatch_config_github.yaml`, rename it to `sbatch_config.yaml` and adjust the values to your setup.

Copy the `template_sbatch_nodes_github.sh`, rename it to `template_sbatch_nodes.sh` and adjust the values to your setup.


## Start Runs

You can start runs with the `main_train.py` file. For example

You can queue up runs in SLURM environments by running `python main_sbatch.py --hp <YAML> --time <TIME> --nodes <NODES>`
which will queue up a run that uses the hyperparameters from `<YAML>` and queues up a run on `<NODES>` nodes.


## Run

All hyperparameters have to be defined in a yaml file that is passed via the `--hp <YAML>` CLI argument.
You can start runs on "normal" servers or SLURM environments.

### Run on "Normal" Servers

Define how many (and which) GPUs you want to use with the `--devices` CLI argument
- `--devices 0` will start the run on the GPU with index 0
- `--devices 2` will start the run on the GPU with index 2
- `--devices 0,1,2,3` will start the run on 4 GPUs

Examples:
- `python main_train.py --devices 0,1,2,3 --hp yamls/stage2/l16_mae.yaml`
- `python main_train.py --devices 0,1,2,3,4,5,6,7 --hp yamls/stage3/l16_mae.yaml`

### Run with SLURM

To start runs in SLURM environments, you need to setup the configurations for SLURM as outlined above.
Then start runs with the `main_sbatch.py` script.

Example:
- `python main_sbatch.py --time 24:00:00 --nodes 4 --hp yamls/stage3/l16_mae.yaml`

### Run many yamls

You can run many yamls by creating a folder `yamls_run`, copying all yamls that you want to run
into that folder and then running `python main_run_folder.py --devices 0 --folder yamls_run`.

#### Resume run

Add these flags to your `python main_train.py` or `python main_sbatch.py` command to resume from a checkpoint.

- `--resume_stage_id <STAGGE_ID>` resume from `cp=latest`
- `--resume_stage_id <STAGGE_ID> --resume_checkpoint E100` resume from epoch 100
- `--resume_stage_id <STAGGE_ID> --resume_checkpoint U100` resume from update 100
- `--resume_stage_id <STAGGE_ID> --resume_checkpoint S1024` resume from sample 1024

#### Via yaml
Add a resume initializer to the trainer

```
trainer:
  ...
  initializer:
    kind: resume_initializer
    stage_id: ???
    checkpoint:
      epoch: 100
```