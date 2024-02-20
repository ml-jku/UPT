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

class LagrangianLargeTSimformerTrainer(SgdTrainer):
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
        dataset, collator = self.data_container.get_dataset("train", mode="target_vel_large_t")
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
        return "x curr_pos curr_pos_full edge_index edge_index_target timestep target_vel_large_t target_acc all_pos all_vel target_pos target_pos_encode perm"
    
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
            target_vel_large_t = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="target_vel_large_t", batch=batch)
            target_vel_large_t = target_vel_large_t.to(self.model.device, non_blocking=True)
            curr_pos = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="curr_pos", batch=batch)
            curr_pos = curr_pos.to(self.model.device, non_blocking=True)
            target_pos = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="target_pos", batch=batch)
            target_pos = target_pos.to(self.model.device, non_blocking=True)
            target_pos_encode = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="target_pos_encode", batch=batch)
            target_pos_encode = target_pos_encode.to(self.model.device, non_blocking=True)
            all_pos = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="all_pos", batch=batch)
            all_pos = all_pos.to(self.model.device, non_blocking=True)
            all_vel = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="all_vel", batch=batch)
            all_vel = all_vel.to(self.model.device, non_blocking=True)
            perm = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="perm", batch=batch)
            perm = perm.to(self.model.device, non_blocking=True)

            edge_index = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="edge_index", batch=batch)
            edge_index = edge_index.to(self.model.device, non_blocking=True)
            edge_index_target = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="edge_index_target", batch=batch)
            edge_index_target = edge_index_target.to(self.model.device, non_blocking=True)
            batch_idx = ctx["batch_idx"].to(self.model.device, non_blocking=True)
            unbatch_idx = ctx["unbatch_idx"].to(self.model.device, non_blocking=True)
            unbatch_select = ctx["unbatch_select"].to(self.model.device, non_blocking=True)

            n_input_timesteps = x.shape[1]

            # Flatten input
            x = einops.rearrange(
                x,
                "a num_input_timesteps dim -> a (num_input_timesteps dim)",
            )
            # Targets are predicted for all particles of all batches simultaneously
            target_vel_large_t = einops.rearrange(
                target_vel_large_t,
                "bs n_particles b -> (bs n_particles) b",
            )
            # Get current position for decoding
            prev_pos_decode = all_pos[:,n_input_timesteps,:,:]

            # forward pass
            model_outputs = self.model.forward_large_t(
                x,
                timestep=timestep,
                curr_pos=curr_pos,
                curr_pos_decode=target_pos,
                prev_pos_decode=prev_pos_decode,
                edge_index=edge_index,
                batch_idx=batch_idx,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select,
                edge_index_target = edge_index_target,
                target_pos_encode=target_pos_encode,
                perm_batch=perm,
                **self.trainer.forward_kwargs,
            )

            # next timestep loss
            losses = dict(
                target=self.trainer.loss_function(
                    prediction=model_outputs["target"],
                    target=target_vel_large_t,
                    reduction=reduction,
                ),
            )

            if "prev_target" in model_outputs:
                # Prev target are the first n_input_timesteps timesteps
                target = einops.rearrange(
                    all_vel[:,:n_input_timesteps,:,:],
                    "bs time n_particles dim -> (bs n_particles) (time dim)"
                )
                prev_target_loss = self.trainer.loss_function(
                    prediction=model_outputs["prev_target"],
                    target=target,
                    reduction=reduction,
                )
                losses["prev_target_loss"] = prev_target_loss

            if "pred_dynamics" in model_outputs:
                dynamics_loss = self.trainer.loss_function(
                    prediction=model_outputs["pred_dynamics"],
                    target=model_outputs["dynamics"],
                    reduction='mean',
                )
                losses["dynamics_loss"] = dynamics_loss

            total_loss = losses["target"]
            if "prev_target" in model_outputs:
                total_loss = total_loss + losses["prev_target_loss"]

            if "pred_dynamics" in model_outputs:
                total_loss = total_loss + losses["dynamics_loss"]

            infos = {
                # calculate degree of graph (average number of connections p)
                "degree/input": len(edge_index) / len(x)
            }
            return dict(total=total_loss, **losses), infos
