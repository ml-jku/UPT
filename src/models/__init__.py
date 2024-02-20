import logging
from copy import deepcopy
from functools import partial

import torch.nn as nn
import yaml

from initializers import initializer_from_kwargs
from utils.factory import instantiate


def model_from_kwargs(kind=None, path_provider=None, data_container=None, **kwargs):
    # exclude update_counter from copying (otherwise model and trainer have different update_counter objects)
    update_counter = kwargs.pop("update_counter", None)
    static_ctx = kwargs.pop("static_ctx", None)
    dynamic_ctx = kwargs.pop("dynamic_ctx", None)
    kwargs = deepcopy(kwargs)

    # allow setting multiple kwargs in yaml; but allow also overwriting it
    # kind: vit.masked_encoder
    # kwargs: ${select:${vars.encoder_model_key}:${yaml:models/vit}}
    # patch_size: [128, 1] # this will overwrite the patch_size in kwargs
    kwargs_from_yaml = kwargs.pop("kwargs", {})
    kwargs = {**kwargs_from_yaml, **kwargs}

    # try to load kwargs from checkpoint
    if "initializers" in kwargs:
        # only first one can have use_checkpoint_kwargs
        initializer_kwargs = kwargs["initializers"][0]
        assert all(obj.get("use_checkpoint_kwargs", None) is None for obj in kwargs["initializers"][1:])
        use_checkpoint_kwargs = initializer_kwargs.pop("use_checkpoint_kwargs", False)
        initializer = initializer_from_kwargs(**initializer_kwargs, path_provider=path_provider)
        if use_checkpoint_kwargs:
            ckpt_kwargs = initializer.get_model_kwargs()
            if kind is None and "kind" in ckpt_kwargs:
                kind = ckpt_kwargs.pop("kind")
            else:
                ckpt_kwargs.pop("kind", None)
            # initializer/optim/freezers shouldnt be used
            ckpt_kwargs.pop("initializers", None)
            ckpt_kwargs.pop("optim_ctor", None)
            # check if keys overlap; this can be intended
            # - vit trained with drop_path_rate but then for evaluation this should be set to 0
            # if keys overlap the explicitly specified value dominates (i.e. from yaml or from code)
            kwargs_intersection = set(kwargs.keys()).intersection(set(ckpt_kwargs.keys()))
            if len(kwargs_intersection) > 0:
                logging.info(f"checkpoint_kwargs overlap with kwargs (intersection={kwargs_intersection})")
                for intersecting_kwarg in kwargs_intersection:
                    ckpt_kwargs.pop(intersecting_kwarg)
            kwargs.update(ckpt_kwargs)
            # LEGACY start: shape was stored as torch.Size -> yaml cant parse that
            if "input_shape" in kwargs and not isinstance(kwargs["input_shape"], tuple):
                kwargs["input_shape"] = tuple(kwargs["input_shape"])
            # LEGACY end
            logging.info(f"postprocessed checkpoint kwargs:\n{yaml.safe_dump(kwargs, sort_keys=False)[:-1]}")
        else:
            logging.info(f"not loading checkpoint kwargs")
    else:
        logging.info(f"model has no initializers -> not loading a checkpoint or an optimizer state")

    assert kind is not None, "model has no kind (maybe use_checkpoint_kwargs=True is missing in the initializer?)"

    # rename optim to optim_ctor (in yaml it is intuitive to call it optim as the yaml should not bother with the
    # implementation details but the implementation passes a ctor so it should also be called like it)
    optim = kwargs.pop("optim", None)
    # model doesn't need to have an optimizer
    if optim is not None:
        kwargs["optim_ctor"] = optim

    # filter out modules passed to ctor
    ctor_kwargs_filtered = {k: v for k, v in kwargs.items() if not isinstance(v, nn.Module)}
    ctor_kwargs = deepcopy(ctor_kwargs_filtered)
    ctor_kwargs["kind"] = kind
    ctor_kwargs.pop("input_shape", None)
    ctor_kwargs.pop("output_shape", None)
    ctor_kwargs.pop("optim_ctor", None)

    return instantiate(
        module_names=[
            f"models.{kind}",
            f"models.composite.{kind}",
        ],
        type_names=[kind.split(".")[-1]],
        update_counter=update_counter,
        path_provider=path_provider,
        data_container=data_container,
        static_ctx=static_ctx,
        dynamic_ctx=dynamic_ctx,
        ctor_kwargs=ctor_kwargs,
        **kwargs,
    )


def prepare_momentum_kwargs(kwargs):
    # remove optim from all SingleModels (e.g. used for EMA)
    kwargs = deepcopy(kwargs)
    _prepare_momentum_kwargs(kwargs)
    return kwargs


def _prepare_momentum_kwargs(kwargs):
    if isinstance(kwargs, dict):
        kwargs.pop("optim", None)
        kwargs.pop("freezers", None)
        kwargs.pop("initializers", None)
        kwargs.pop("is_frozen", None)
        for v in kwargs.values():
            _prepare_momentum_kwargs(v)
    elif isinstance(kwargs, partial):
        kwargs.keywords.pop("optim_ctor", None)
        kwargs.keywords.pop("freezers", None)
        kwargs.keywords.pop("initializers", None)
        kwargs.keywords.pop("is_frozen", None)
        for v in kwargs.keywords.values():
            _prepare_momentum_kwargs(v)
