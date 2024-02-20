import logging

import einops
import torch
from torch.utils.data import DataLoader

from distributed.config import get_world_size
from utils.functional import get_powers_of_two, is_power_of_two


def calculate_effective_batch_size_per_device(effective_batch_size, world_size=None):
    world_size = world_size or get_world_size()
    assert effective_batch_size % world_size == 0, \
        f"effective_batch_size ({effective_batch_size}) needs to be multiple of world_size ({world_size})"
    return int(effective_batch_size / world_size)


def calculate_batch_size_and_accumulation_steps(effective_batch_size_per_device, max_batch_size=None):
    # calculate batch_size and accumulation_steps
    if max_batch_size is None:
        batch_size = effective_batch_size_per_device
        accumulation_steps = 1
    else:
        if effective_batch_size_per_device <= max_batch_size:
            # fits into memory
            batch_size = effective_batch_size_per_device
            accumulation_steps = 1
        else:
            # multiple accumulation steps
            msg = "effective_batch_size_per_device needs to be multiple of max_batch_size"
            assert effective_batch_size_per_device % max_batch_size == 0, msg
            accumulation_steps = int(effective_batch_size_per_device / max_batch_size)
            batch_size = int(effective_batch_size_per_device / accumulation_steps)
    return batch_size, accumulation_steps


def calculate_automatic_max_batch_size(
        train_dataset,
        train_step_fn,
        effective_batch_size_per_device,
        device,
        model,
        collator=None,
):
    if str(device) == "cpu":
        return effective_batch_size_per_device
    # batchsizes that are not a power of two are not supported
    if not is_power_of_two(effective_batch_size_per_device):
        return effective_batch_size_per_device

    # backup state_dict (state_dict doesn't clone tensors -> call .clone on every tensor in the state dict)
    model_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    optim_state_dicts = {}
    for name, submodel in model.submodels.items():
        if submodel.optim is None:
            continue
        sd = submodel.optim.state_dict()
        cloned = {}
        for key in sd.keys():
            if key == "state":
                cloned["state"] = {
                    idx_key: {k: v.clone() if v is not None else v for k, v in idx_dict.items()}
                    for idx_key, idx_dict in sd["state"].items()
                }
            elif key == "param_groups":
                cloned["param_groups"] = [{k: v for k, v in group.items()} for group in sd["param_groups"]]
            elif key == "param_idx_to_name":
                cloned["param_idx_to_name"] = {k: v for k, v in sd["param_idx_to_name"].items()}
            else:
                raise NotImplementedError
        optim_state_dicts[name] = cloned

    # compose batch_sizes to try (start from 2 because some models do batchnorm during training [e.g. barlow twins])
    batch_sizes = get_powers_of_two(2, effective_batch_size_per_device)

    # make a train_step with decreasing batch_sizes (faster when batchsize is actually correct)
    # NOTE: this makes runs only deterministic per device if a stochastic transformation is used
    #   as one sample is loaded from the dataset with its stochastic transforms and the stochastic transforms
    #   have no seed the random generator of stochastic transforms will be progressed
    sample, sample_ctx = next(iter(DataLoader(train_dataset, batch_size=1, collate_fn=collator)))
    max_batch_size = 1
    for batch_size in reversed(batch_sizes):
        logging.info(f"trying batch_size {batch_size}")

        # scale batch_size by repeating the sample
        if isinstance(sample, (list, tuple)):
            data = []
            for item in sample:
                if isinstance(item, (list, tuple)):
                    data.append([einops.repeat(entry, "1 ... -> bs ...", bs=batch_size) for entry in item])
                else:
                    data.append(einops.repeat(item, "1 ... -> bs ...", bs=batch_size))
        else:
            data = einops.repeat(sample, "1 ... -> bs ...", bs=batch_size)
        # wrap into tuple
        if isinstance(data, list):
            data = tuple(data)
        # scale batch_size of ctx
        ctx = {
            k: einops.repeat(v, "1 ... -> bs ...", bs=batch_size) if torch.is_tensor(v) else v
            for k, v in sample_ctx.items()
        }

        # try update step
        try:
            train_step_fn(batch=(data, ctx))
            max_batch_size = batch_size
            break
        except RuntimeError as e:
            if not str(e).startswith("CUDA out of memory"):
                raise e
            model.clear_buffers()

    # restore state_dict
    model.load_state_dict(model_state_dict)
    for name, submodel in model.submodels.items():
        if submodel.optim is None:
            continue
        submodel.optim.load_state_dict(optim_state_dicts[name])
    # clear buffers if models track something during the forward pass --> e.g. NnclrQueue
    model.clear_buffers()
    return max_batch_size
