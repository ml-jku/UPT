import matplotlib.pyplot as plt
import os
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import io

from datasets.collators.cfd_simformer_collator import CfdSimformerCollator
import scipy
from functools import partial

import einops
import torch
from kappadata.wrappers import ModeWrapper
from kappautils.images.png import png_writer_viridis
from torchvision.datasets.folder import default_loader

from callbacks.base.periodic_callback import PeriodicCallback
from utils.formatting_util import dict_to_string
from kappautils.images.points_to_image import coords_to_image
from utils.param_checking import to_2tuple

class OfflineRolloutMeshCallback(PeriodicCallback):
    def __init__(
            self,
            dataset_key,
            num_rollout_timesteps=None,
            use_teacher_forcing=False,
            rollout_kwargs=None,
            resolution=None,
            save_gif=False,
            save_pngs=False,
            save_plots=False,
            visualize_pressure=False,
            visualize_velocities=False,
            visualize_velocity_magnitude=True,
            duration_per_frame=100,
            visualization_backend="torch",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        # properties that are initialized in before_training
        self.__config_id = None
        self.out = None
        self.dataset = None
        self.num_rollout_timesteps = num_rollout_timesteps
        self.use_teacher_forcing = use_teacher_forcing
        self.rollout_kwargs = rollout_kwargs or {}
        self.resolution = resolution
        # what to save (gif and/or png)
        self.save_gif = save_gif
        self.save_pngs = save_pngs
        self.save_plots = save_plots
        # what to visualize (pressure and/or seperate velocities and/or velocity magnitude)
        self.visualize_pressure = visualize_pressure
        self.visualize_velocities = visualize_velocities
        self.visualize_velocity_magnitude = visualize_velocity_magnitude
        # visualization params
        self.duration_per_frame = duration_per_frame
        self.visualization_backend = visualization_backend

    def _before_training(self, **kwargs):
        if os.name == "nt" and "KMP_DUPLICATE_LIB_OK" not in os.environ:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.out = self.path_provider.stage_output_path / "rollout"
        self.out.mkdir(exist_ok=True)
        (self.out / "gifs").mkdir(exist_ok=True)
        if self.save_pngs:
            (self.out / "pngs").mkdir(exist_ok=True)
        if self.save_plots:
            (self.out / "plots").mkdir(exist_ok=True)
        self.dataset, collator = self.data_container.get_dataset(key=self.dataset_key, mode=self.dataset_mode)
        assert isinstance(collator.collator, CfdSimformerCollator)
        # how many timesteps to roll out?
        if self.num_rollout_timesteps is None:
            self.num_rollout_timesteps = self.dataset.getdim_timestep()
        else:
            assert 0 < self.num_rollout_timesteps <= self.dataset.getdim_timestep()

    @property
    def dataset_mode(self):
        return "index pos edge_index x geometry2d velocity"

    def _register_sampler_configs(self, trainer):
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode=self.dataset_mode)

    @staticmethod
    def _tensor_to_pil_matplotlib(data, progress, pos):
        y, x = pos.cpu().unbind(1)
        data = data.cpu()
        # convert points (3, num_points) to image (3, height, width)
        with io.BytesIO() as buffer:
            images = []
            for i in range(3):
                buffer.seek(0)
                plt.scatter(x, y, s=0.01, c=data[i].cpu(), cmap="viridis")
                plt.axis("off")
                plt.xlim(0, 300)
                plt.ylim(0, 200)
                plt.savefig(buffer, bbox_inches="tight", format="jpg")
                images.append(to_tensor(Image.open(buffer)))
        data = torch.concat(images, dim=1).to(pos.device)

        # add a progress line on top
        progress_tensor = torch.zeros(size=(data.size(0), 1, data.size(2)), dtype=data.dtype, device=data.device)
        progress_tensor[:, :, :round(progress * data.size(2))] = 1
        data = torch.concat([progress_tensor, data], dim=1)
        return to_pil_image(data)

    def _tensor_to_pil_torch(self, data, progress, pos):
        # convert points (3, num_points) to image (3, height, width)
        data = torch.stack([
            coords_to_image(
                coords=pos,
                resolution=self.resolution,
                weights=data[i],
            )
            for i in range(3)
        ])

        # normalize to [0, 1]
        # data has shape (3, height, width)
        data_min = data.flatten(start_dim=1).min(dim=1).values
        data -= data_min.view(-1, 1, 1)
        data_max = data.flatten(start_dim=1).max(dim=1).values
        data /= data_max.view(-1, 1, 1)

        # stack images ontop of each other (ground_truth, prediction, delta)
        data = einops.rearrange(data, "three height width -> (three height) width")

        # add a progress line on top
        progress_tensor = torch.zeros(size=(1, data.size(1),), dtype=data.dtype, device=data.device)
        progress_tensor[:, :round(progress * data.size(1))] = 1
        data = torch.concat([progress_tensor, data])

        # to image
        temp_out = self.path_provider.get_temp_path() / f"{self.path_provider.stage_id}.png"
        png_writer_viridis(data.unsqueeze(0), temp_out)
        pil = default_loader(temp_out)

        return pil

    def tensor_to_pil(self, data, progress, pos):
        if self.visualization_backend == "matplotlib":
            return self._tensor_to_pil_matplotlib(data=data, progress=progress, pos=pos)
        if self.visualization_backend == "torch":
            return self._tensor_to_pil_torch(data=data, progress=progress, pos=pos)
        raise NotImplementedError

    def visualize(self, idx, trajectories, deltas, pos):
        if sum([self.visualize_pressure, self.visualize_velocities, self.visualize_velocity_magnitude]) == 0:
            return
        if sum([self.save_gif, self.save_pngs]) == 0:
            return

        assert idx.numel() == 1, "only batchsize=1 is supported for now"

        # concatenate prediction with delta
        # (2 * batch_size * num_points, num_rollout_timesteps, num_channels) ->
        # (3 * batch_size * num_points, num_rollout_timesteps, num_channels)
        trajectories = torch.concat([trajectories, deltas])

        # generate gifs + images
        for i, trajectory in enumerate([trajectories]):
            prefix = f"{self.dataset_key}_{self.update_counter.cur_checkpoint}_{self.visualization_backend}"
            if len(self.rollout_kwargs) > 0:
                prefix = f"{prefix}_{dict_to_string(self.rollout_kwargs, item_seperator='-')}"
            if self.use_teacher_forcing:
                prefix = f"{prefix}_tforced"
            prefix = f"{prefix}_idx{idx[i]:04d}"
            data = {}
            # what to visualize (pressure, velocity magnitude, seperate velocities)
            if self.visualize_pressure:
                data["pressure"] = trajectory[:, :, 0]
            if self.visualize_velocities:
                data["v0"] = trajectory[:, :, 1]
                data["v1"] = trajectory[:, :, 2]
            if self.visualize_velocity_magnitude:
                velocity = trajectory[:, :, 1:]
                velocity_magnitude = torch.sqrt(torch.sum(velocity ** 2, dim=2))
                data["vmag"] = velocity_magnitude

            for name, item in data.items():
                # generate images
                self.logger.info(f"generating {name} images")
                # data has shape (3 * height, width) -> normalize each sub-image seperatle
                item = einops.rearrange(
                    item,
                    "(three num_points) num_rollout_timesteps -> num_rollout_timesteps three num_points",
                    three=3,
                )
                imgs = [
                    self.tensor_to_pil(
                        item[j],
                        progress=j / max(1, (len(item) - 1)),
                        pos=pos,
                    )
                    for j in range(len(item))
                ]
                if self.save_gif:
                    uri = self.out / "gifs" / f"{name}_{prefix}.gif"
                    self.logger.info(f"generating {name} gif '{uri.as_posix()}'")
                    imgs[0].save(
                        fp=uri,
                        format="GIF",
                        append_images=imgs[1:],
                        save_all=True,
                        duration=self.duration_per_frame,
                        loop=0,
                    )
                if self.save_pngs:
                    self.logger.info(f"storing individual {name} pngs")
                    for j, img in enumerate(imgs):
                        img.save(self.out / "pngs" / f"{name}_{prefix}_ts{j:04d}.png")

    def _forward(self, batch, model, trainer):
        # prepare data
        batch, ctx = batch
        idx = ModeWrapper.get_item(mode=self.dataset_mode, item="index", batch=batch)
        x = ModeWrapper.get_item(mode=self.dataset_mode, item="x", batch=batch)
        geometry2d = ModeWrapper.get_item(mode=self.dataset_mode, item="geometry2d", batch=batch)
        geometry2d = geometry2d.to(model.device, non_blocking=True)
        velocity = ModeWrapper.get_item(mode=self.dataset_mode, item="velocity", batch=batch)
        velocity = velocity.to(model.device, non_blocking=True)
        pos = ModeWrapper.get_item(mode=self.dataset_mode, item="pos", batch=batch)
        pos = pos.to(model.device, non_blocking=True)
        padded_pos = ctx["padded_pos"].to(model.device, non_blocking=True)
        batch_idx = ctx["batch_idx"].to(model.device, non_blocking=True)
        unbatch_idx = ctx["unbatch_idx"].to(model.device, non_blocking=True)
        unbatch_select = ctx["unbatch_select"].to(model.device, non_blocking=True)
        edge_index = ModeWrapper.get_item(mode=self.dataset_mode, item="edge_index", batch=batch)
        edge_index = edge_index.to(model.device, non_blocking=True)
        assert x.ndim == 3, "expected data to be of shape (bs * num_points, num_total_timesteps + 1, num_channels)"
        if x.size(1) != self.num_rollout_timesteps + 1:
            x = x[:, :self.num_rollout_timesteps + 1]
        x = x.to(model.device, non_blocking=True)

        # rollout
        with trainer.autocast_context:
            if self.use_teacher_forcing:
                assert self.num_rollout_timesteps + 1 == x.size(1)
                predictions = model.rollout_teacher_forced(
                    x=x,
                    geometry2d=geometry2d,
                    velocity=velocity,
                    pos=pos,
                    padded_pos=padded_pos,
                    batch_idx=batch_idx,
                    unbatch_idx=unbatch_idx,
                    unbatch_select=unbatch_select,
                    edge_index=edge_index,
                    num_rollout_timesteps=self.num_rollout_timesteps,
                    **self.rollout_kwargs,
                )
            else:
                predictions = model.rollout(
                    x0=x[:, 0],
                    geometry2d=geometry2d,
                    velocity=velocity,
                    pos=pos,
                    padded_pos=padded_pos,
                    batch_idx=batch_idx,
                    unbatch_idx=unbatch_idx,
                    unbatch_select=unbatch_select,
                    edge_index=edge_index,
                    num_rollout_timesteps=self.num_rollout_timesteps,
                    **self.rollout_kwargs,
                )

        # ground truth excludes t0
        ground_truth = x[:, 1:1 + self.num_rollout_timesteps]

        # concatenate prediction with ground truth (along height dimension)
        # (batch_size * num_points, num_rollout_timesteps, num_channels) ->
        # (2 * batch_size * num_points, num_rollout_timesteps, num_channels)
        trajectories = torch.concat([ground_truth, predictions])

        # calculate normalized normalized_deltas
        # dont calculate to save memory
        # normalized_deltas = (ground_truth - predictions).abs()

        # free memory
        del ground_truth
        del predictions

        # denormalize (from mean=0 std=1 to original value range)
        trajectories = einops.rearrange(
            trajectories,
            "num_points num_timesteps num_channels -> num_timesteps num_channels num_points",
        )
        trajectories = self.dataset.denormalize(trajectories, inplace=True)
        trajectories = einops.rearrange(
            trajectories,
            "num_timesteps num_channels num_points -> num_points num_timesteps num_channels",
        )

        # calculate denormalized delta
        denormed_ground_truth, denormed_predictions = trajectories.chunk(2)
        denormalized_deltas = (denormed_ground_truth - denormed_predictions).abs()

        # calculate movement: i.e. how much changes between timesteps (\hat{x}_t - \hat{x}_{t-1})
        # denormed_movement = (denormed_predictions - denormed_predictions.roll(shifts=(-1,), dims=(1,)))[:, :-1]
        # denormed_movement = denormed_movement.abs()

        # generate visualizations
        self.visualize(idx=idx, trajectories=trajectories, deltas=denormalized_deltas, pos=pos)

        # calculate deltas ("losses")
        results = dict(
            # overall_normalized_delta=normalized_deltas.flatten(start_dim=1).mean(dim=-1),
            overall_denormalized_delta=denormalized_deltas.flatten(start_dim=1).mean(dim=-1),
            # overall_denormalized_movement=denormed_movement.flatten(start_dim=1).mean(dim=1),
        )
        if self.save_plots:
            # results.update(
            #     delta_per_channel=einops.rearrange(
            #         normalized_deltas,
            #         "bs num_rollout_steps num_channels ... -> bs num_channels (num_rollout_steps ...)"
            #     ).mean(dim=-1),
            #     delta_per_timestep=einops.rearrange(
            #         normalized_deltas,
            #         "bs num_rollout_steps num_channels ... -> bs num_rollout_steps (num_channels ...)"
            #     ).mean(dim=-1),
            #     delta_per_channel_per_timestep=einops.rearrange(
            #         normalized_deltas,
            #         "bs num_rollout_steps num_channels ... -> bs num_rollout_steps num_channels (...)"
            #     ).mean(dim=-1),
            # )
            raise NotImplementedError
        return results

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, batch_size, data_iter, **_):
        results = self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )

        # log deltas
        metric_identifier = f"{self.dataset_key}/0to{self.num_rollout_timesteps}"
        # file_identifier = f"{metric_identifier.replace('/', '_')}_{self.update_counter.cur_checkpoint}"
        if len(self.rollout_kwargs) > 0:
            metric_identifier = f"{metric_identifier}/{dict_to_string(self.rollout_kwargs)}"
            # file_identifier = f"{file_identifier}_{dict_to_string(self.rollout_kwargs, item_seperator='_')}"
        if self.use_teacher_forcing:
            metric_identifier = f"{metric_identifier}/tforced"
            # file_identifier = f"{file_identifier}_tforced"
        # overall
        # self.writer.add_scalar(
        #     key=f"delta/{metric_identifier}/overall/normalized",
        #     value=results["overall_normalized_delta"].mean(),
        #     logger=self.logger,
        #     format_str=".10f",
        # )
        self.writer.add_scalar(
            key=f"delta/{metric_identifier}/overall/denormalized",
            value=results["overall_denormalized_delta"].mean(),
            logger=self.logger,
            format_str=".10f",
        )
        # self.writer.add_scalar(
        #     key=f"movement/{metric_identifier}/overall/denormalized",
        #     value=results["overall_denormalized_movement"].mean(),
        #     logger=self.logger,
        #     format_str=".10f",
        # )
        # plots
        if self.save_plots:
            # torch.save(
            #     results["delta_per_channel"].mean(dim=0),
            #     self.out / "plots" / f"PerChannel_{file_identifier}.th",
            # )
            # torch.save(
            #     results["delta_per_timestep"].mean(dim=0),
            #     self.out / "plots" / f"PerTimestep_{file_identifier}.th",
            # )
            # torch.save(
            #     results["delta_per_channel_per_timestep"].mean(dim=0),
            #     self.out / "plots" / f"PerChannelPerTimestep_{file_identifier}.th",
            # )
            raise NotImplementedError
