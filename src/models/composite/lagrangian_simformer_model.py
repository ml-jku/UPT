import einops
import torch
import torch.nn.functional as F

from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create


class LagrangianSimformerModel(CompositeModelBase):
    def __init__(
            self,
            encoder,
            latent,
            decoder,
            conditioner=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        common_kwargs = dict(
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )
        # timestep embed
        self.conditioner = create(
            conditioner,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.input_shape,
        )
        # set static_ctx["dim"]
        if self.conditioner is not None:
            self.static_ctx["dim"] = self.conditioner.dim
        else:
            self.static_ctx["dim"] = latent["kwargs"]["dim"]
        # encoder
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            **common_kwargs,
        )
        assert self.encoder.output_shape is not None
        # dynamics
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
        # Box for PBC 
        self.box = self.data_container.get_dataset().box

    @property
    def submodels(self):
        return dict(
            **(dict(conditioner_encoder=self.conditioner) if self.conditioner is not None else {}),
            encoder=self.encoder,
            latent=self.latent,
            decoder=self.decoder,
        )

    # noinspection PyMethodOverriding
    def forward(
            self,
            x,
            timestep,
            curr_pos,
            curr_pos_decode,
            prev_pos_decode,
            edge_index,
            batch_idx,
            edge_index_target=None,
            target_pos_encode=None,
            perm_batch=None,
            unbatch_idx=None,
            unbatch_select=None,
            reconstruct_prev_target=False,
            encode_target=False,
            predict_velocity=False
    ):
        outputs = {}

        # encode timestep t
        if self.conditioner is not None:
            # No velocity for lagrangian simulations -> set to 0
            timestep_embed = self.conditioner(timestep=timestep, velocity=torch.zeros_like(timestep))
        else:
            timestep_embed = None

        # encode data (v_{t-T}, ..., v{t} -> dynamic_{t-1})
        prev_dynamics = self.encoder(
            x,
            mesh_pos=curr_pos,
            mesh_edges=edge_index,
            batch_idx=batch_idx,
            condition=timestep_embed
        )
        outputs["prev_dynamics"] = prev_dynamics

        # predict current latent (dynamic_{t-1} -> dynamic_t)
        dynamics = self.latent(
            prev_dynamics,
            condition=timestep_embed
        )
        outputs["dynamics"] = dynamics

        # decode next_latent to next_data (dynamic_t -> target)
        target = self.decoder(
            dynamics,
            query_pos=curr_pos_decode,
            unbatch_idx=unbatch_idx,
            unbatch_select=unbatch_select,
            condition=timestep_embed
        )
        outputs["target"] = target

        # reconstruct a_{t} from dynamic_{t-1}
        if reconstruct_prev_target:
            if self.conditioner is not None:
                prev_timestep_embed = self.conditioner(timestep=timestep-1, velocity=torch.zeros_like(timestep))
            else:
                prev_timestep_embed = None
            # reconstruct prev_target
            prev_target = self.decoder(
                prev_dynamics,
                query_pos=prev_pos_decode,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select,
                condition=prev_timestep_embed
            )
            outputs["prev_target"] = prev_target

        # Reconstruct input from dynamic
        if encode_target:
            if self.conditioner is not None:
                next_timestep_embed = self.conditioner(timestep=timestep+1, velocity=torch.zeros_like(timestep))
            else:
                next_timestep_embed = None
            vels = einops.rearrange(x,
                    "bs_times_n_particles (timesteps dim) -> bs_times_n_particles timesteps dim", 
                    dim=target.shape[1]
                )
            if predict_velocity:
                current_vel = target[perm_batch]
            else:
                # Get last velocity
                current_vel = vels[:,-1,:]
                # Unnormalize for integration later to get positions
                current_vel = self.data_container.get_dataset().unnormalize_vel(current_vel)
                # Get acceleration -> target only at perm_batch and unnormalized
                a = self.data_container.get_dataset().unnormalize_acc(target[perm_batch])
                # Integrate
                current_vel = current_vel + a
                # Normalize velocity
                current_vel = self.data_container.get_dataset().normalize_vel(current_vel)
            # Add new velocity to input of the encoder
            new_vels = torch.concat((vels[:,1:,:], current_vel.unsqueeze(dim=1)), dim=1)
            new_vels = einops.rearrange(new_vels,
                        "bs_times_n_particles timesteps dim -> bs_times_n_particles (timesteps dim)")
            pred_dynamics = self.encoder(
                new_vels,
                mesh_pos=target_pos_encode,
                mesh_edges=edge_index_target,
                batch_idx=batch_idx,
                condition=next_timestep_embed
            )
            outputs["pred_dynamics"] = pred_dynamics

        return outputs
    
    # noinspection PyMethodOverriding
    def forward_large_t(
            self,
            x,
            timestep,
            curr_pos,
            curr_pos_decode,
            prev_pos_decode,
            edge_index,
            batch_idx,
            edge_index_target=None,
            target_pos_encode=None,
            perm_batch=None,
            unbatch_idx=None,
            unbatch_select=None,
            reconstruct_prev_target=False,
            encode_target=False,
            const_timestep=False
    ):
        outputs = {}

        # encode timestep t
        # No velocity for lagrangian simulations -> set to 0
        if self.conditioner is not None:
            if const_timestep:
                # Constant timestep over all time
                timestep = torch.tensor([0]).to(x.device)
                timestep_embed = self.conditioner(timestep=timestep, velocity=torch.zeros_like(timestep))
                next_timestep_embed = timestep_embed
            else:
                timestep_embed = self.conditioner(timestep=timestep, velocity=torch.zeros_like(timestep))
                next_timestep = timestep + self.data_container.get_dataset().n_pushforward_timesteps + 1
                next_timestep_embed = self.conditioner(timestep=next_timestep, velocity=torch.zeros_like(timestep))
        else:
            timestep_embed = None
            next_timestep_embed = None

        # encode data (v_{t-1}, v{t} -> dynamic_{t-1})
        prev_dynamics = self.encoder(
            x,
            mesh_pos=curr_pos,
            mesh_edges=edge_index,
            batch_idx=batch_idx,
            condition=timestep_embed
        )
        outputs["prev_dynamics"] = prev_dynamics

        # predict current latent (dynamic_{t-1} -> dynamic_t)
        dynamics = self.latent(
            prev_dynamics,
            condition=timestep_embed
        )
        outputs["dynamics"] = dynamics

        # decode next_latent to next_data (dynamic_t -> target)
        target = self.decoder(
            dynamics,
            query_pos=curr_pos_decode,
            unbatch_idx=unbatch_idx,
            unbatch_select=unbatch_select,
            condition=next_timestep_embed
        )
        outputs["target"] = target

        # reconstruct prev_target from dynamic_{t-1}
        if reconstruct_prev_target:
            # reconstruct prev_x_hat
            prev_target = self.decoder(
                prev_dynamics,
                query_pos=prev_pos_decode,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select,
                condition=timestep_embed
            )
            outputs["prev_target"] = prev_target

        if encode_target:
            pred_dynamics = self.encoder(
                target[perm_batch],
                mesh_pos=target_pos_encode,
                mesh_edges=edge_index_target,
                batch_idx=batch_idx,
                condition=next_timestep_embed
            )
            outputs["pred_dynamics"] = pred_dynamics

        return outputs

    @torch.no_grad()
    def rollout(
            self,
            x,
            timestep,
            curr_pos,
            edge_index,
            batch_idx,
            unbatch_idx=None,
            unbatch_select=None,
            full_rollout=False,
            rollout_length=20,
            predict_velocity=False
    ):
        if self.conditioner is not None:
            timestep_embed = self.conditioner(timestep=timestep, velocity=torch.zeros_like(timestep))
        else:
            timestep_embed = None

        # encode data (v_{t-T}, ..., v{t} -> dynamic_{t-1})
        dynamics = self.encoder(
            x,
            mesh_pos=curr_pos,
            mesh_edges=edge_index,
            batch_idx=batch_idx,
            condition=timestep_embed
        )
        vels = einops.rearrange(x,
                "bs_times_n_particles (timesteps dim) -> bs_times_n_particles timesteps dim", 
                dim=curr_pos.shape[1]
            )
        # Get last velocity
        current_vel = vels[:,-1,:]
        # Unnormalize for integration later to get positions
        current_vel = self.data_container.get_dataset().unnormalize_vel(current_vel)
        all_predictions = []
        all_velocities = []
        curr_pos = einops.rearrange(
            curr_pos,
            "(bs n_particles) dim -> bs n_particles dim",
            bs=len(unbatch_select)
        )
        
        for i in range(rollout_length):
            # predict current latent (dynamic_{t-1} -> dynamic_t)
            dynamics = self.latent(
                dynamics,
                condition=timestep_embed
            )
            # decode next_latent to next_data (dynamic_t -> a_{t+1})
            target = self.decoder(
                dynamics,
                query_pos=curr_pos,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select,
                condition=timestep_embed
            )
            if predict_velocity:
                current_vel = self.data_container.get_dataset().unnormalize_vel(target)
            else:
                # Unnormalize a_hat to calculate next velocity
                a_hat = self.data_container.get_dataset().unnormalize_acc(target)
                # Calculate new velocity
                current_vel = current_vel + a_hat
            # Unscale curr_pos to be in original scale
            curr_pos = self.data_container.get_dataset().unscale_pos(curr_pos)
            curr_pos = einops.rearrange(
                curr_pos,
                "bs n_particles n_dim -> (bs n_particles) n_dim",
            )
            # Integrate
            curr_pos = (curr_pos + current_vel) % self.box.to(curr_pos.device)
            # New position
            curr_pos = einops.rearrange(
                curr_pos,
                "(bs n_particles) n_dim -> bs n_particles n_dim", bs=len(unbatch_select)
            )
            curr_vel_reshaped = einops.rearrange(
                current_vel,
                "(bs n_particles) n_dim -> bs n_particles n_dim", bs=len(unbatch_select)
            )
            all_predictions.append(curr_pos)
            all_velocities.append(curr_vel_reshaped)
            # Scale new position for decoder
            curr_pos = self.data_container.get_dataset().scale_pos(curr_pos)
            # New timestep embedding
            timestep = timestep + 1
            if self.conditioner is not None:
                timestep_embed = self.conditioner(timestep=timestep, velocity=torch.zeros_like(timestep))
            else:
                timestep_embed = None
            
            if full_rollout:
                # Normalize current vel to be used as input for the encoder
                current_vel_normalized = self.data_container.get_dataset().normalize_vel(current_vel)
                x = torch.concat([vels[:,1:,:], current_vel_normalized.unsqueeze(1)],dim=1)
                x = einops.rearrange(
                    x,
                    "bs num_input_timesteps num_points -> bs (num_input_timesteps num_points)",
                )
                mesh_pos = einops.rearrange(
                    curr_pos,
                    "bs n_particles dim -> (bs n_particles) dim",
                )
                dynamics = self.encoder(
                    x,
                    mesh_pos=mesh_pos,
                    mesh_edges=edge_index,
                    batch_idx=batch_idx,
                    condition=timestep_embed
                )
                
        all_predictions = torch.stack(all_predictions, dim=1)
        all_velocities = torch.stack(all_velocities, dim=1)
        return all_predictions, all_velocities
    
    @torch.no_grad()
    def rollout_large_t(
            self,
            x,
            all_pos,
            timestep,
            edge_index,
            batch_idx,
            unbatch_idx=None,
            unbatch_select=None
    ):
        pos_idx = self.data_container.get_dataset().n_input_timesteps - 1
        large_t = self.data_container.get_dataset().n_pushforward_timesteps + 1

        # timestep = torch.tensor([pos_idx])
        if self.conditioner is not None:
            if timestep is not None:
                timestep_embed = self.conditioner(timestep=timestep, velocity=torch.zeros_like(timestep))
                const_timestep = False
            else:
                timestep = torch.tensor([0]).to(x.device)
                timestep_embed = self.conditioner(timestep=timestep, velocity=torch.zeros_like(timestep))
                const_timestep = True
        else:
            timestep_embed = None

        curr_pos = all_pos[:,pos_idx,:,:]
        curr_pos = einops.rearrange(
            curr_pos,
            "bs n_particles dim -> (bs n_particles) dim",
        )
        # encode data 
        dynamics = self.encoder(
            x,
            mesh_pos=curr_pos,
            mesh_edges=edge_index,
            batch_idx=batch_idx,
            condition=timestep_embed
        )
        vel_predictions = []
        while True:
            # predict current latent
            dynamics = self.latent(
                dynamics,
                condition=timestep_embed
            )
            # Next position index
            pos_idx = pos_idx + large_t
            # Check if still in trajectory
            if pos_idx >= all_pos.shape[1]:
                break
            curr_pos = all_pos[:,pos_idx,:,:]
            # New timestep embedding
            
            if self.conditioner is not None and not const_timestep:
                timestep = timestep + large_t
                timestep_embed = self.conditioner(timestep=timestep, velocity=torch.zeros_like(timestep))
                
            # decode next_latent to next_data (dynamic_t -> a_{t+1})
            v_hat = self.decoder(
                dynamics,
                query_pos=curr_pos,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select,
                condition=timestep_embed
            )
            # Reshape to get time dim
            v_hat = einops.rearrange(
                v_hat,
                "a (time dim) -> a time dim",
                dim=curr_pos.shape[-1]
            )
            vel_predictions.append(v_hat)
                
        vel_predictions = torch.concat(vel_predictions, dim=1)
        return vel_predictions
    