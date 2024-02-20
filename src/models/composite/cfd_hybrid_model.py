import torch

from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create


class CfdHybridModel(CompositeModelBase):
    def __init__(
            self,
            encoder,
            latent,
            decoder,
            conditioner=None,
            force_latent_fp32=True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.force_latent_fp32 = force_latent_fp32
        common_kwargs = dict(
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )
        # conditioner
        self.conditioner = create(
            conditioner,
            model_from_kwargs,
            **common_kwargs,
        )
        # encoder
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            **common_kwargs,
        )
        # latent
        self.latent = create(
            latent,
            model_from_kwargs,
            input_shape=self.encoder.output_shape,
            **common_kwargs,
        )
        # decoder
        self.decoder = create(
            decoder,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.latent.output_shape,
            output_shape=self.output_shape,
        )

    @property
    def submodels(self):
        return dict(
            **(dict(conditioner=self.conditioner) if self.conditioner is not None else {}),
            encoder=self.encoder,
            latent=self.latent,
            decoder=self.decoder,
        )

    # noinspection PyMethodOverriding
    def forward(
            self,
            x,
            mesh_pos,
            grid_pos,
            query_pos,
            mesh_to_grid_edges,
            unbatch_idx,
            unbatch_select,
            timestep=None,
            velocity=None,
    ):
        outputs = {}

        if self.conditioner is not None:
            condition = self.conditioner(timestep=timestep, velocity=velocity)
        else:
            condition = None

        # encode data ((x_{t-2}, x_{t-1} -> dynamic_{t-1})
        prev_dynamics = self.encoder(
            x,
            mesh_pos=mesh_pos,
            grid_pos=grid_pos,
            mesh_to_grid_edges=mesh_to_grid_edges,
            condition=condition,
        )

        # predict current latent (dynamic_{t-1} -> dynamic_t)
        if self.force_latent_fp32:
            with torch.autocast(device_type=str(x.device).split(":")[0], enabled=False):
                prev_dynamics = prev_dynamics.float()
                condition = condition.float()
                dynamics = self.latent(prev_dynamics, condition=condition)
        else:
            dynamics = self.latent(prev_dynamics, condition=condition)

        # decode next_latent to next_data (dynamic_t -> x_t)
        x_hat = self.decoder(
            dynamics,
            query_pos=query_pos,
            unbatch_idx=unbatch_idx,
            unbatch_select=unbatch_select,
            condition=condition,
        )
        outputs["x_hat"] = x_hat

        return outputs

    @torch.no_grad()
    def rollout(
            self,
            x,
            mesh_pos,
            grid_pos,
            query_pos,
            mesh_to_grid_edges,
            unbatch_idx,
            unbatch_select,
            velocity=None,
            num_rollout_timesteps=None,
    ):
        # check num_rollout_timesteps
        max_timesteps = self.data_container.get_dataset().getdim_timestep()
        num_rollout_timesteps = num_rollout_timesteps or max_timesteps
        assert 0 < num_rollout_timesteps <= max_timesteps
        # setup
        x_hats = []
        timestep = torch.zeros(1, device=x.device, dtype=torch.long)
        for _ in range(num_rollout_timesteps):
            # predict next timestep
            outputs = self(
                x,
                mesh_pos=mesh_pos,
                grid_pos=grid_pos,
                query_pos=query_pos,
                mesh_to_grid_edges=mesh_to_grid_edges,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select,
                timestep=timestep,
                velocity=velocity,
            )
            x_hat = outputs["x_hat"]
            x_hats.append(x_hat)
            # shift last prediction into history
            x = torch.concat([x[:, x_hat.size(1):], x_hat], dim=1)
            # increase timestep
            timestep.add_(1)

        # num_rollout_timesteps * (batch_size * num_points, num_channels)
        # -> (batch_size * num_points, num_channels, num_rollout_timesteps)
        return torch.stack(x_hats, dim=2)
