from copy import deepcopy
from functools import partial

from optimizers.interleaved_optimizer import InterleavedOptimizer
from optimizers.optimizer_wrapper import OptimizerWrapper
from utils.factory import get_ctor


def optim_ctor_from_kwargs(kind, **kwargs):
    kwargs = deepcopy(kwargs)
    if kind == "interleaved_optimizer":
        optim_ctors = [optim_ctor_from_kwargs(**optim) for optim in kwargs.pop("optims")]
        return partial(InterleavedOptimizer, optim_ctors=optim_ctors, **kwargs)

    # extract optimizer wrapper kwargs
    wrapped_optim_kwargs = {}
    wrapped_optim_kwargs_keys = [
        "schedule",
        "weight_decay_schedule",
        "clip_grad_value",
        "clip_grad_norm",
        "exclude_bias_from_wd",
        "exclude_norm_from_wd",
        "param_group_modifiers",
        "lr_scaler",
    ]
    for key in wrapped_optim_kwargs_keys:
        if key in kwargs:
            wrapped_optim_kwargs[key] = kwargs.pop(key)

    torch_optim_ctor = get_ctor(
        module_names=["torch.optim", f"optimizers.custom.{kind}"],
        type_names=[kind],
        **kwargs,
    )
    return partial(_optimizer_wrapper_ctor, torch_optim_ctor=torch_optim_ctor, **wrapped_optim_kwargs)


def _optimizer_wrapper_ctor(model, torch_optim_ctor, **wrapped_optim_kwargs):
    return OptimizerWrapper(model=model, torch_optim_ctor=torch_optim_ctor, **wrapped_optim_kwargs)
