import kappamodules.utils.tensor_cache as tc
import os
from functools import cached_property
import torch
import einops
from torch import nn
from kappadata.wrappers import ModeWrapper

from losses import loss_fn_from_kwargs
from utils.factory import create
from .base.sgd_trainer import SgdTrainer
from datasets.collators.lagrangian_simformer_collator import LagrangianSimformerCollator
from functools import cached_property
from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback

class LagrangianSimformerTrainer(SgdTrainer):
    def __init__(
            self,
            loss_function,
            forward_kwargs=None,
            max_batch_size=None,
            **kwargs
    ):
        # automatic batchsize is not supported with mesh data
        disable_gradient_accumulation = max_batch_size is None
        super().__init__(
            max_batch_size=max_batch_size,
            disable_gradient_accumulation=disable_gradient_accumulation,
            **kwargs,
        )
        self.loss_function = create(loss_function, loss_fn_from_kwargs, update_counter=self.update_counter)
        self.forward_kwargs = forward_kwargs or {}

    def get_trainer_callbacks(self, model=None):
        return [
            UpdateOutputCallback(
                keys=["degree/input"],
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
            UpdateOutputCallback(
                keys=["degree/input"],
                **self.get_default_callback_intervals(),
                **self.get_default_callback_kwargs(),
            ),
        ]

    @cached_property
    def input_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="x")
        sample, _ = dataset[0]
        # num_input_timesteps are concated along channel dimension
        input_shape = list(sample.shape[1:])
        input_shape[0] *= sample.size(0)
        if collator is not None:
            assert isinstance(collator.collator, LagrangianSimformerCollator)
            assert len(input_shape) == 2
            input_shape[1] = None
        self.logger.info(f"input_shape: {tuple(input_shape)}")
        return tuple(input_shape)

    @cached_property
    def output_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="target_acc")
        sample, _ = dataset[0]
        output_shape = list(sample.shape)[::-1]
        if collator is not None:
            assert isinstance(collator.collator, LagrangianSimformerCollator)
            assert len(output_shape) == 2
            output_shape[1] = None
        self.logger.info(f"output_shape: {tuple(output_shape)}")
        return tuple(output_shape)

    @cached_property
    def dataset_mode(self):
        return "x curr_pos curr_pos_full edge_index edge_index_target timestep target_acc target_pos prev_pos prev_acc target_pos_encode perm target_vel"
    
    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(nn.Module):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        def forward(self, batch, reduction="mean"):
            # prepare data
            batch, ctx = batch
            x = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="x", batch=batch)
            x = x.to(self.model.device, non_blocking=True)
            timestep = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="timestep", batch=batch)
            timestep = timestep.to(self.model.device, non_blocking=True)
            target_acc = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="target_acc", batch=batch)
            target_acc = target_acc.to(self.model.device, non_blocking=True)
            target_vel = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="target_vel", batch=batch)
            target_vel = target_vel.to(self.model.device, non_blocking=True)
            curr_pos = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="curr_pos", batch=batch)
            curr_pos = curr_pos.to(self.model.device, non_blocking=True)
            curr_pos_full = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="curr_pos_full", batch=batch)
            curr_pos_full = curr_pos_full.to(self.model.device, non_blocking=True)
            prev_pos = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="prev_pos", batch=batch)
            prev_pos = prev_pos.to(self.model.device, non_blocking=True)
            prev_acc = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="prev_acc", batch=batch)
            prev_acc = prev_acc.to(self.model.device, non_blocking=True)
            edge_index = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="edge_index", batch=batch)
            edge_index = edge_index.to(self.model.device, non_blocking=True)
            batch_idx = ctx["batch_idx"].to(self.model.device, non_blocking=True)
            unbatch_idx = ctx["unbatch_idx"].to(self.model.device, non_blocking=True)
            unbatch_select = ctx["unbatch_select"].to(self.model.device, non_blocking=True)

            target_pos_encode = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="target_pos_encode", batch=batch)
            target_pos_encode = target_pos_encode.to(self.model.device, non_blocking=True)
            perm = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="perm", batch=batch)
            perm = perm.to(self.model.device, non_blocking=True)
            edge_index_target = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="edge_index_target", batch=batch)
            edge_index_target = edge_index_target.to(self.model.device, non_blocking=True)

            # Flatten input
            x = einops.rearrange(
                x, 
                "bs num_input_timesteps num_points -> bs (num_input_timesteps num_points)",
            )
            # Targets are predicted for all particles of all batches simultaneously
            target_acc = einops.rearrange(
                target_acc,
                "bs n_particles n_dim -> (bs n_particles) n_dim",
            )
            target_vel = einops.rearrange(
                target_vel,
                "bs n_particles n_dim -> (bs n_particles) n_dim",
            )
            prev_acc = einops.rearrange(
                prev_acc,
                "bs n_particles n_dim -> (bs n_particles) n_dim",
            )

            # forward pass
            model_outputs = self.model(
                x,
                timestep=timestep,
                curr_pos=curr_pos,
                curr_pos_decode=curr_pos_full,
                prev_pos_decode=prev_pos,
                edge_index=edge_index,
                batch_idx=batch_idx,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select,
                edge_index_target = edge_index_target,
                target_pos_encode=target_pos_encode,
                perm_batch=perm,
                **self.trainer.forward_kwargs,
            )

            # Target is either next velocity or next acceleration
            if self.trainer.forward_kwargs['predict_velocity']:
                target = target_vel
            else:
                target = target_acc

            # next timestep loss
            losses = dict(
                target=self.trainer.loss_function(
                    prediction=model_outputs["target"],
                    target=target,
                    reduction=reduction,
                ),
            )
            # input_reconstruction losses
            if "prev_target" in model_outputs:
                prev_target_loss = self.trainer.loss_function(
                    prediction=model_outputs["prev_target"],
                    target=prev_acc,
                    reduction=reduction,
                )
                losses["prev_target"] = prev_target_loss

            if "pred_dynamics" in model_outputs:
                pred_dynamics_loss = self.trainer.loss_function(
                    prediction=model_outputs["pred_dynamics"],
                    target=model_outputs["dynamics"],
                    reduction=reduction,
                )
                losses["pred_dynamics"] = pred_dynamics_loss
            # weight losses                   
            total_loss = losses["target"]
            if "prev_target" in losses:
                total_loss = total_loss + losses["prev_target"]
            if "pred_dynamics" in losses:
                total_loss = total_loss + losses["pred_dynamics"]

            infos = {
                # calculate degree of graph (average number of connections p)
                "degree/input": len(edge_index) / len(x)
            }

            return dict(total=total_loss, **losses), infos
