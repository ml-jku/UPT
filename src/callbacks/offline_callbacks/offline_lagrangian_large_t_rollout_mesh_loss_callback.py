from torch_geometric.utils import scatter
from kappadata.wrappers import ModeWrapper
from functools import partial

import einops
import torch

from callbacks.base.periodic_callback import PeriodicCallback
from utils.formatting_util import dict_to_string


class OfflineLagrangianLargeTRolloutMeshLossCallback(PeriodicCallback):
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

        # Get index for all timesteps which are predicted
        n_pushforward_timesteps = self.data_container.get_dataset().n_pushforward_timesteps
        n_vels_traj = self.data_container.get_dataset().n_seq - 1
        large_t = n_pushforward_timesteps + 1
        time_indicies = [(i, i+1) for i in range(large_t, n_vels_traj-2, large_t)]
        self.time_indicies = torch.tensor([item for sublist in time_indicies for item in sublist])


    def _before_training(self, trainer, **kwargs):
        self.out.mkdir(exist_ok=True)
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
        # x is needed to encode the first latent
        x = ModeWrapper.get_item(mode=trainer.dataset_mode, item="x", batch=batch)
        x = x.to(model.device, non_blocking=True)
        # all positions of the sequence are needed for decoding
        all_pos = ModeWrapper.get_item(mode=trainer.dataset_mode, item="all_pos", batch=batch)
        all_pos = all_pos.to(model.device, non_blocking=True)
        # all velocities are needed to compare the predictions
        all_vel = ModeWrapper.get_item(mode=trainer.dataset_mode, item="all_vel", batch=batch)
        all_vel = all_vel.to(model.device, non_blocking=True)
        # get the timestep
        if 'const_timestep' in trainer.forward_kwargs and trainer.forward_kwargs['const_timestep']:
            timestep = None
        else:
            timestep = ModeWrapper.get_item(mode=trainer.dataset_mode, item="timestep", batch=batch)
            timestep = timestep.to(model.device, non_blocking=True)

        edge_index = ModeWrapper.get_item(mode=trainer.dataset_mode, item="edge_index", batch=batch)
        edge_index = edge_index.to(model.device, non_blocking=True)
        batch_idx = ctx["batch_idx"].to(model.device, non_blocking=True)

        # Flatten input
        x = einops.rearrange(
            x,
            "a num_input_timesteps dim -> a (num_input_timesteps dim)",
        )

        unbatch_idx = ctx["unbatch_idx"].to(model.device, non_blocking=True)
        unbatch_select = ctx["unbatch_select"].to(model.device, non_blocking=True)

        # rollout
        with trainer.autocast_context:
            vel_predictions = model.rollout_large_t(
                x=x,
                all_pos=all_pos,
                timestep=timestep,
                edge_index=edge_index,
                batch_idx=batch_idx,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select
            )

        # Prepare target
        
        all_vel = einops.rearrange(
            all_vel,
            "bs time n_particles dim -> (bs n_particles) time dim"
        )
        all_vel_target = all_vel[:,self.time_indicies,:]
        dt = self.data_container.get_dataset().metadata["dt"] * self.data_container.get_dataset().metadata["write_every"]
        dx = self.data_container.get_dataset().metadata["dx"]
        dim = self.data_container.get_dataset().metadata["dim"]
        # Unnormalize velocity
        all_vel_target = self.data_container.get_dataset().unnormalize_vel(all_vel_target)
        all_vel_target = all_vel_target
        vel_predictions = self.data_container.get_dataset().unnormalize_vel(vel_predictions)
        vel_predictions = vel_predictions
        # Unbatch
        all_vel_target = einops.rearrange(
            all_vel_target,
            "(bs n_particles) time dim -> bs n_particles time dim",
            bs=len(unbatch_select)
        )
        vel_predictions = einops.rearrange(
            vel_predictions,
            "(bs n_particles) time dim -> bs n_particles time dim",
            bs=len(unbatch_select)
        )
        # Calculate ekin like in lagrangebench
        ekin_predictions = ((vel_predictions / dt) ** 2).sum(dim=(1,3))
        ekin_predictions = ekin_predictions * dx**dim

        ekin_target = ((all_vel_target / dt) ** 2).sum(dim=(1,3))
        ekin_target = ekin_target * dx**dim

        diff_norm = (vel_predictions - all_vel_target).norm(dim=3).mean(dim=(1,2))
        relative_norm = ((vel_predictions - all_vel_target).norm(dim=3) / all_vel_target.norm(dim=3)).mean(dim=(1,2))

        results_dict = {
            "predicted": ekin_predictions.mean(dim=1),
            "target": ekin_target.mean(dim=1),
            "mse": ((ekin_predictions - ekin_target) ** 2).mean(),
            "vel_error": diff_norm,
            "vel_error_relative": relative_norm
        }

        if self.rollout_kwargs['save_rollout']:
            rollout_dict = {'ekin_target': ekin_target,
                            'ekin_predictions': ekin_predictions,
                            'vel_target': all_vel_target,
                            'vel_predictions': vel_predictions,
                            'traj_idx': ctx['traj_idx'],
                            'time_idx': self.time_indicies}
            
            outpath = self.out / f"rollout_results_{str(self.update_counter.cur_checkpoint).lower()}.pt"
            torch.save(rollout_dict, outpath)

        return results_dict

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, trainer_model, batch_size, data_iter, **_):
        results = self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer, trainer_model=trainer_model),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )
        # log deltas
        for key in results.keys():
            metric_identifier = f"{self.dataset_key}/{key}"
            if len(self.rollout_kwargs) > 0:
                metric_identifier = f"{metric_identifier}"
            self.writer.add_scalar(
                key=f"ekin/{metric_identifier}",
                value=results[key].mean(),
                logger=self.logger,
                format_str=".10f",
            )