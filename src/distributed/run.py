import logging
import os
import platform

import psutil
import torch
import yaml
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.multiprocessing import spawn

from .config import (
    is_managed,
    get_world_size_from_env,
    get_rank_from_env,
    get_local_rank,
    is_custom_managed_run,
    is_mpi_managed_run,
    get_nodes,
)


def run_managed(accelerator, devices, main_single):
    assert is_managed()
    if accelerator == "gpu":
        # custom managed run doesn't set CUDA_VISIBLE_DEVICES
        if is_custom_managed_run() or is_mpi_managed_run() or len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(get_local_rank())
        _check_single_device_visible()
    if devices is None:
        world_size = get_world_size_from_env()
        if world_size == 1:
            _run_managed_singleprocess(accelerator, main_single)
        else:
            # use all GPUs for training
            _run_managed_multiprocess(accelerator, main_single)
    else:
        # use single GPU (e.g. run_folder from every GPU)
        world_size, device_ids = _parse_devices(accelerator, devices)
        assert world_size == 1 and len(device_ids) == 1
        _log_device_info(accelerator, device_ids)
        _run_managed_singleprocess(accelerator, main_single)


def _run_managed_singleprocess(accelerator, main_single):
    # single process
    logging.info(f"running single process slurm training")
    device = _accelerator_to_device(accelerator)
    main_single(device=device)


def _run_managed_multiprocess(accelerator, main_single):
    # setup MASTER_ADDR & MASTER_PORT
    assert "MASTER_ADDR" in os.environ
    assert "MASTER_PORT" in os.environ

    # get config from env variables
    world_size = get_world_size_from_env()
    rank = get_rank_from_env()

    # init process group
    logging.info(
        f"initializing rank={rank} local_rank={get_local_rank()} "
        f"nodes={get_nodes()} hostname={platform.uname().node} "
        f"master_addr={os.environ['MASTER_ADDR']} master_port={os.environ['MASTER_PORT']} "
        f"(waiting for all {world_size} processes to connect)"
    )
    init_process_group(backend=get_backend(accelerator), init_method="env://", world_size=world_size, rank=rank)
    barrier()

    # start main_single
    device = _accelerator_to_device(accelerator)
    main_single(device=device)
    destroy_process_group()


def run_single_or_multiprocess(accelerator, devices, main_single, master_port, mig_devices):
    logging.info("------------------")
    # single node run
    assert devices is not None
    world_size, device_ids = _parse_devices(accelerator, devices, mig_devices)
    if world_size == 1:
        # single process
        logging.info(f"running single process training")
        if accelerator == "gpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[0]
            _check_single_device_visible()
        _log_device_info(accelerator, device_ids)
        device = _accelerator_to_device(accelerator)
        main_single(device=device)
    else:
        # spawn multi process training
        logging.info(
            f"running multi process training on {world_size} processes "
            f"(devices={devices} host={platform.uname().node})"
        )
        master_port = _get_free_port(master_port)
        logging.info(f"master port: {master_port}")
        # dont log device info as this would load torch on device0 and block the VRAM required for this
        # log_device_info(accelerator, device_ids)
        args = (accelerator, device_ids, master_port, world_size, main_single)
        spawn(_run_multiprocess, nprocs=world_size, args=args)


def _run_multiprocess(rank, accelerator, device_ids, master_port, world_size, main_single):
    # currently only single node is supported
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    if accelerator == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[rank]
        _check_single_device_visible()

    from torch.distributed import init_process_group, destroy_process_group
    init_process_group(
        backend=get_backend(accelerator, device_ids),
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    device = _accelerator_to_device(accelerator)
    main_single(device=device)
    destroy_process_group()


def get_backend(accelerator, device_ids=None):
    if accelerator == "cpu":
        # gloo is recommended for cpu multiprocessing
        # https://pytorch.org/docs/stable/distributed.html#which-backend-to-use
        return "gloo"
    if os.name == "nt":
        # windows doesn't support nccl (I think)
        return "gloo"
    # MIG doesn't support NCCL
    if device_ids is not None:
        for device_id in device_ids:
            try:
                int(device_id)
            except ValueError:
                return "gloo"
    # nccl is recommended for gpu multiprocessing
    # https://pytorch.org/docs/stable/distributed.html#which-backend-to-use
    return "nccl"


def _get_free_port(start_port):
    taken_ports = set()
    for connection in psutil.net_connections():
        if connection.laddr.ip == "127.0.0.1":
            taken_ports.add(connection.laddr.port)
        if len(connection.raddr) > 0 and connection.raddr.ip == "127.0.0.1":
            taken_ports.add(connection.raddr.port)

    for port in range(start_port, 65535):
        if port not in taken_ports:
            return port
    raise ValueError(f"all ports starting from {start_port} are taken")


def _parse_devices(accelerator, devices, mig_devices=None):
    try:
        # single process
        device_ids = [int(devices)]
    except ValueError:
        # multi process
        device_ids = yaml.safe_load(f"[{devices}]")
        msg = f"invalid devices specification '{devices}' (specify multiple devices like this '0,1,2,3')"
        assert all(isinstance(d, int) for d in device_ids), msg
    # os.environ["CUDA_VISIBLE_DEVICES"] requires string
    device_ids = [str(device_id) for device_id in device_ids]

    if accelerator == "gpu" and mig_devices is not None:
        # map to MIG device ids
        hostname = platform.uname().node
        if hostname in mig_devices:
            for i in range(len(device_ids)):
                device_id = int(device_ids[i])
                if device_id in mig_devices[hostname]:
                    mig_device_id = mig_devices[hostname][device_id]
                    device_ids[i] = mig_device_id
                    logging.info(f"device_id is MIG device with id {mig_device_id}")

    return len(device_ids), device_ids


def _check_single_device_visible():
    assert "CUDA_VISIBLE_DEVICES" in os.environ
    visible_device_count = torch.cuda.device_count()
    assert visible_device_count <= 1, os.environ


def _log_device_info(accelerator, device_ids):
    if accelerator == "cpu":
        for i in range(len(device_ids)):
            logging.info(f"device {i}: cpu")
    elif accelerator == "gpu":
        # retrieve device names via nvidia-smi because CUDA_VISIBLE_DEVICES needs to be set before calling anything
        # in torch.cuda -> only 1 visible device
        all_devices = os.popen("nvidia-smi --query-gpu=gpu_name --format=csv,noheader").read().strip().split("\n")
        for i, device_id in enumerate(device_ids):
            try:
                device_id = int(device_id)
                logging.info(f"device {i}: {all_devices[device_id]} (id={device_id})")
            except ValueError:
                # MIG device
                logging.info(f"using MIG device")
    else:
        raise NotImplementedError


def _accelerator_to_device(accelerator):
    if accelerator == "cpu":
        return "cpu"
    elif accelerator == "gpu":
        return "cuda"
    raise NotImplementedError
