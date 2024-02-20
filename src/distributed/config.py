import logging
import os

import torch.distributed as dist


def is_slurm_run():
    if os.environ.get("SLURM_JOB_NAME", None) == "interactive":
        return False
    return "SLURM_PROCID" in os.environ and "SLURM_NTASKS_PER_NODE" in os.environ


def is_mpi_managed_run():
    return (
            "OMPI_COMM_WORLD_SIZE" in os.environ and
            "OMPI_COMM_WORLD_RANK" in os.environ and
            "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ and
            "OMPI_MCA_orte_num_nodes" in os.environ
    )


def is_custom_managed_run():
    return (
            "CUSTOM_NUM_NODES" in os.environ and
            "CUSTOM_WORLD_SIZE" in os.environ and
            "CUSTOM_RANK" in os.environ and
            "CUSTOM_LOCAL_RANK" in os.environ
    )


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_local_rank():
    if get_nodes() == 1 and os.environ.get("SLURM_TASKS_PER_NODE") == "1":
        return get_rank()
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    if "CUSTOM_LOCAL_RANK" in os.environ:
        return int(os.environ["CUSTOM_LOCAL_RANK"])
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    return get_rank()


def is_data_rank0():
    # data has to be copied in 2 cases
    # - is_local_rank0: single-gpu, multi-gpu, multi-gpu SLURM
    #   - process with is_local_rank0 copies the data
    #   - other processes have to wait for the copying to finish via barrier
    # - get_world_size == 1: SLURM runs that are not using multi-gpu require every process to copy data
    #   - no guarantee that the processes use the same dataset
    #   - avoid race conditions
    return is_local_rank0() or get_world_size() == 1


def is_managed():
    return is_slurm_run() or is_custom_managed_run() or is_mpi_managed_run()


def get_nodes():
    if "SLURM_JOB_NUM_NODES" in os.environ:
        return int(os.environ["SLURM_JOB_NUM_NODES"])
    if "CUSTOM_NUM_NODES" in os.environ:
        return int(os.environ["CUSTOM_NUM_NODES"])
    if "OMPI_MCA_orte_num_nodes" in os.environ:
        return int(os.environ["OMPI_MCA_orte_num_nodes"])
    return 1


def get_world_size_from_env():
    if "SLURM_NTASKS_PER_NODE" in os.environ:
        return get_nodes() * int(os.environ["SLURM_NTASKS_PER_NODE"])
    if "CUSTOM_WORLD_SIZE" in os.environ:
        return int(os.environ["CUSTOM_WORLD_SIZE"])
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])
    raise NotImplementedError


def get_rank_from_env():
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    if "CUSTOM_RANK" in os.environ:
        return int(os.environ["CUSTOM_RANK"])
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_RANK"])
    raise NotImplementedError


def is_rank0():
    return get_rank() == 0


def is_local_rank0():
    return get_local_rank() == 0


def barrier():
    if is_distributed():
        dist.barrier()


def is_own_work(idx):
    return idx % get_world_size() == get_rank()


def get_backend():
    if is_distributed():
        return dist.get_backend()
    return None


def log_distributed_config():
    logging.info("------------------")
    logging.info("DIST CONFIG")
    logging.info(f"rank: {get_rank()}")
    logging.info(f"local_rank: {get_local_rank()}")
    logging.info(f"world_size: {get_world_size()}")
    logging.info(f"nodes: {get_nodes()}")
    logging.info(f"backend: {get_backend()}")
    if "SLURM_JOB_ID" in os.environ:
        logging.info(f"slurm job id: {os.environ['SLURM_JOB_ID']}")
    if "PBS_JOBID" in os.environ:
        logging.info(f"pbs job id: {os.environ['PBS_JOBID']}")
