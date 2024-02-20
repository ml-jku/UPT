from collections import defaultdict

import numpy as np
import torch

from callbacks.base.periodic_callback import PeriodicCallback
from distributed.gather import all_reduce_mean_grad


class NumSupernodesCallback(PeriodicCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_first_log = True
        self.trainer_batch_size = None
        self.hook = None
        self.num_nodes_history = []
        self.num_supernodes_history = []

    class NumSupernodesHook:
        def __init__(self):
            self.num_nodes = None
            self.num_supernodes = None
            self.enabled = False

        def __call__(self, module, module_input, module_output):
            if self.enabled:
                if self.num_nodes is None:
                    self.num_nodes = len(module_input[0])
                if self.num_supernodes is None:
                    self.num_supernodes = len(module_output[0])

    def _before_training(self, trainer_batch_size, model, **kwargs):
        self.trainer_batch_size = trainer_batch_size
        if hasattr(model.encoder, "mesh_embed"):
            if hasattr(model.encoder.mesh_embed, "pool"):
                self.hook = self.NumSupernodesHook()
                model.encoder.mesh_embed.pool.register_forward_hook(self.hook)

    def before_every_accumulation_step(self, **kwargs):
        if self.hook is None:
            return
        self.hook.enabled = True

    def _track_after_accumulation_step(self, **kwargs):
        if self.hook is None:
            return
        self.hook.enabled = False
        self.num_nodes_history.append(self.hook.num_nodes)
        self.num_supernodes_history.append(self.hook.num_supernodes)
        self.hook.num_nodes = None
        self.hook.num_supernodes = None
        if self.is_first_log:
            self.logger.info(
                f"num_nodes: per_device={self.num_nodes_history[-1]} "
                f"per_sample={self.num_nodes_history[-1] // self.trainer_batch_size}"
            )
            self.logger.info(
                f"num_supernodes: per_device={self.num_supernodes_history[-1]} "
                f"per_sample={self.num_supernodes_history[-1] // self.trainer_batch_size}"
            )
            self.is_first_log = False

    def _periodic_callback(self, **_):
        if self.hook is None:
            return
        # log averages
        self.writer.add_scalar(
            key=f"num_nodes/{self.to_short_interval_string()}",
            value=float(np.mean(self.num_nodes_history) / self.trainer_batch_size),
        )
        self.writer.add_scalar(
            key=f"num_supernodes/{self.to_short_interval_string()}",
            value=float(np.mean(self.num_supernodes_history) / self.trainer_batch_size),
        )
        self.num_nodes_history.clear()
        self.num_supernodes_history.clear()
