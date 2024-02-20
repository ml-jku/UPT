import gc

import torch


def get_tensors_in_memory():
    # some warning was thrown when calling torch.is_tensor(_reduce_op) with a _reduce_op object
    all_objs = gc.get_objects()
    all_tensors = []
    cuda_tensors = []
    for obj in all_objs:
        try:
            if type(obj).__name__ != "_reduce_op" and torch.is_tensor(obj):
                all_tensors.append(obj)
                if obj.device != torch.device("cpu"):
                    cuda_tensors.append(obj)
        except ReferenceError:
            # with wandb there is some issue where 'ReferenceError: weakly-referenced object no longer exists' is raised
            pass
    return all_tensors, cuda_tensors
