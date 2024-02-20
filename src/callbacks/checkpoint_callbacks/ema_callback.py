from collections import defaultdict

import torch

from callbacks.base.periodic_callback import PeriodicCallback
from distributed.config import is_rank0
from utils.select_with_path import select_with_path


class EmaCallback(PeriodicCallback):
    def __init__(self, target_factors, model_paths=None, **kwargs):
        super().__init__(**kwargs)
        self.model_paths = model_paths or [None]
        self.target_factors = target_factors
        self.parameters = defaultdict(dict)
        self.buffers = defaultdict(dict)

    def _before_training(self, model, **kwargs):
        if not is_rank0():
            return
        for model_path in self.model_paths:
            cur_model = select_with_path(obj=model, path=model_path)
            for target_factor in self.target_factors:
                for name, param in cur_model.named_parameters():
                    self.parameters[(model_path, target_factor)][name] = param.clone()
            for name, buffer in cur_model.named_buffers():
                self.buffers[model_path][name] = buffer.clone()

    def _track_after_update_step(self, model, **kwargs):
        if not is_rank0():
            return
        for model_path in self.model_paths:
            cur_model = select_with_path(obj=model, path=model_path)
            for target_factor in self.target_factors:
                for name, param in cur_model.named_parameters():
                    key = (model_path, target_factor)
                    self.parameters[key][name].mul_(target_factor).add_(param, alpha=1. - target_factor)
            for name, buffer in cur_model.named_buffers():
                self.buffers[model_path][name].copy_(buffer)

    def _save(self, ckpt, model):
        if not is_rank0():
            return
        for model_path in self.model_paths:
            cur_model = select_with_path(obj=model, path=model_path)
            for target_factor in self.target_factors:
                state_dict = {**self.parameters[(model_path, target_factor)], **self.buffers[model_path]}
                ckpt_dict = dict(
                    state_dict=state_dict,
                    ctor_kwargs=cur_model.ctor_kwargs,
                    ckpt=ckpt,
                    abs_ckpt=dict(self.update_counter.cur_checkpoint),
                    stage_id=self.path_provider.stage_id,
                    ema=target_factor,
                )
                if model_path is None:
                    cur_model_path = model.name
                else:
                    cur_model_path = f"{model.name}.{model_path}"
                fname = f"{cur_model_path} cp={ckpt} ema={target_factor} model.th"
                torch.save(ckpt_dict, self.path_provider.checkpoint_path / fname)

    def _periodic_callback(self, model, **kwargs):
        self._save(ckpt=self.update_counter.cur_checkpoint, model=model)

    def _after_training(self, model, **kwargs):
        self._save(ckpt="last", model=model)
