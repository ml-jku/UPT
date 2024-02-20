#!/bin/bash -l
#SBATCH --account={account}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=<PARTITION>
#SBATCH --qos={qos}
#SBATCH --gres=gpu:4
#SBATCH --time={time}
#SBATCH --chdir={chdir}
#SBATCH --output={output}


### set the first node name as master address
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT={master_port}

# set LD_LIBRARY_PATH (importing torchmetrics imports matplotlib and that fails sometimes if this is not done)
export LD_LIBRARY_PATH={ld_lib_path}:$LD_LIBRARY_PATH

# activate conda env
conda activate {env_name}

# write python command to log file -> easy check for which run crashed if there is some config issue
echo python main_{script}.py {cli_args}

# run
srun --cpus-per-task {cpus_per_task} python main_{script}.py {cli_args}