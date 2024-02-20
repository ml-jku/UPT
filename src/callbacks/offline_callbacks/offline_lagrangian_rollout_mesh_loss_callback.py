from torch_geometric.utils import scatter
from kappadata.wrappers import ModeWrapper
from functools import partial

import einops
import torch

from callbacks.base.periodic_callback import PeriodicCallback
from utils.formatting_util import dict_to_string


class OfflineLagrangianRolloutMeshLossCallback(PeriodicCallback):
    def __init__(
            self,
            dataset_key,
            num_rollout_timesteps=None,
            rollout_kwargs=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.num_rollout_timesteps = num_rollout_timesteps
        self.rollout_kwargs = rollout_kwargs or {}
        self.out = self.path_provider.stage_output_path / "rollout"
        # properties that are initialized in before_training
        self.__config_id = None
        bounds = torch.tensor(self.data_container.get_dataset().metadata['bounds'])
        self.box = bounds[:, 1] - bounds[:, 0]
        if 'full_rollout' in self.rollout_kwargs:
            self.full_rollout = self.rollout_kwargs['full_rollout']
            if self.rollout_kwargs['full_rollout']:
                self.out = self.out / "full_rollout"
            else:
                self.out = self.out / "latent_rollout"
        else:
            self.full_rollout = False

    def _before_training(self, trainer, **kwargs):
        self.out.mkdir(parents=True, exist_ok=True)
        dataset, _ = self.data_container.get_dataset(key=self.dataset_key, mode=trainer.dataset_mode)
        # how many timesteps to roll out?
        if self.num_rollout_timesteps is None:
            self.num_rollout_timesteps = dataset.getdim_timestep()
        else:
            assert 0 < self.num_rollout_timesteps <= dataset.getdim_timestep()

    def _register_sampler_configs(self, trainer):
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode=trainer.dataset_mode)

    def _forward(self, batch, model, trainer, trainer_model):
        # prepare data
        batch, ctx = batch
        x = ModeWrapper.get_item(mode=trainer.dataset_mode, item="x", batch=batch)
        x = x.to(model.device, non_blocking=True)
        timestep = ModeWrapper.get_item(mode=trainer.dataset_mode, item="timestep", batch=batch)
        timestep = timestep.to(model.device, non_blocking=True)
        curr_pos = ModeWrapper.get_item(mode=trainer.dataset_mode, item="curr_pos", batch=batch)
        curr_pos = curr_pos.to(model.device, non_blocking=True)
        target_pos = ModeWrapper.get_item(mode=trainer.dataset_mode, item="target_pos", batch=batch)
        target_pos = target_pos.to(model.device, non_blocking=True)
        edge_index = ModeWrapper.get_item(mode=trainer.dataset_mode, item="edge_index", batch=batch)
        edge_index = edge_index.to(model.device, non_blocking=True)
        batch_idx = ctx["batch_idx"].to(model.device, non_blocking=True)

        # inputs are the velocities of all timesteps
        x = einops.rearrange(
            x,
            "bs num_input_timesteps num_points -> bs (num_input_timesteps num_points)",
        )
        
        # decoder predicts all points
        unbatch_idx = ctx["unbatch_idx"].to(model.device, non_blocking=True)
        unbatch_select = ctx["unbatch_select"].to(model.device, non_blocking=True)

        # rollout
        with trainer.autocast_context:
            x_hat, all_vels = model.rollout(x=x,
                                            timestep=timestep,
                                            curr_pos=curr_pos,
                                            edge_index=edge_index,
                                            batch_idx=batch_idx,
                                            unbatch_idx=unbatch_idx,
                                            unbatch_select=unbatch_select,
                                            full_rollout=self.full_rollout,
                                            rollout_length=target_pos.shape[1],
                                            predict_velocity=trainer.forward_kwargs['predict_velocity']
            )
        vel = (x_hat - target_pos)
        self.box = self.box.to(x_hat.device)
        vel = (vel + self.box * 0.5) % self.box - 0.5 * self.box
        mse = vel ** 2
        mse = mse.mean(dim=[2,3])
        mse2 = mse[:,:2].mean(dim=1)
        mse5 =  mse[:,:5].mean(dim=1)
        mse20 = mse[:,:20].mean(dim=1)

        if self.rollout_kwargs['save_rollout']:
            rollout_dict = {'x_target': target_pos,
                            'x_predictions': x_hat,
                            'v_predictions': all_vels,
                            'traj_idx': ctx['traj_idx']}
            
            outpath = self.out / f"rollout_results_{str(self.update_counter.cur_checkpoint).lower()}.pt"
            torch.save(rollout_dict, outpath)

        return dict(
            mse2=mse2,
            mse5=mse5,
            mse20=mse20,
        )

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, trainer_model, batch_size, data_iter, **_):
        results = self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer, trainer_model=trainer_model),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )
        if self.full_rollout:
            identifier = 'full_rollout'
        else:
            identifier = 'latent_rollout'

        # log deltas
        for num_rollout_timesteps in [2,5,20]:
            metric_identifier = f"{self.dataset_key}/{identifier}/0to{num_rollout_timesteps}"
            if len(self.rollout_kwargs) > 0:
                metric_identifier = f"{metric_identifier}"
            self.writer.add_scalar(
                key=f"delta/{metric_identifier}/overall/mse",
                value=results[f"mse{num_rollout_timesteps}"].mean(),
                logger=self.logger,
                format_str=".10f",
            )