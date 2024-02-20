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
from functools import partial

import einops
import torch

from callbacks.base.periodic_callback import PeriodicCallback
from utils.formatting_util import dict_to_string
from kappadata.wrappers import ModeWrapper

class OfflineRolloutMeshGifCallback(PeriodicCallback):
    def __init__(
            self,
            dataset_key,
            resolution,
            num_rollout_timesteps=None,
            rollout_kwargs=None,
            **kwargs,
    ):
        super().__init__(batch_size=1, **kwargs)
        self.dataset_key = dataset_key
        self.resolution = resolution
        self.num_rollout_timesteps = num_rollout_timesteps
        self.rollout_kwargs = rollout_kwargs or {}
        # properties that are initialized in before_training
        self.__config_id = None
        self.dataset_mode = None
        self.dataset = None
        self.out = None

    def _register_sampler_configs(self, trainer):
        self.dataset_mode = ModeWrapper.add_item(mode=trainer.dataset_mode, item="index")
        self.dataset, _ = self.data_container.get_dataset(key=self.dataset_key, mode=self.dataset_mode)
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode=self.dataset_mode)

    def _before_training(self, trainer, **kwargs):
        self.out = self.path_provider.stage_output_path / "rollout"
        self.out.mkdir(exist_ok=True)
        # how many timesteps to roll out?
        if self.num_rollout_timesteps is None:
            self.num_rollout_timesteps = self.dataset.getdim_timestep()
        else:
            assert 0 < self.num_rollout_timesteps <= self.dataset.getdim_timestep()


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
        with io.BytesIO() as buffer:
            png_writer_viridis(data.unsqueeze(0), buffer, save_format="png")
            buffer.seek(0)
            img = Image.open(buffer)
            pil = img.convert("RGB")
        return pil

    def _forward(self, batch, model, trainer, trainer_model):
        data = trainer_model.prepare(batch, dataset_mode=self.dataset_mode, mode="rollout")
        batch, ctx = batch
        idx = ModeWrapper.get_item(mode=self.dataset_mode, item="index", batch=batch)
        assert "target" not in data
        x = data.pop("x")
        assert x.ndim == 3, "expected data to be of shape (bs * num_points, num_total_timesteps + 1, input_dim)"

        # cut away excess timesteps
        if x.size(1) != self.num_rollout_timesteps + 1:
            x = x[:, :self.num_rollout_timesteps + 1]

        # concat input timesteps
        model_input_dim, _ = model.input_shape
        _, _, x_input_dim = x.shape
        assert model_input_dim % x_input_dim == 0
        num_input_timesteps = model_input_dim // x_input_dim
        x0 = einops.repeat(
            x[:, 0],
            "batch_num_points num_channels ... -> batch_num_points (num_input_timesteps num_channels) ...",
            num_input_timesteps=num_input_timesteps,
        )

        # timestep is manually counted
        data.pop("timestep", None)

        # rollout
        with trainer.autocast_context:
            predictions = model.rollout(
                x0=x0,
                num_rollout_timesteps=self.num_rollout_timesteps,
                **data,
                **self.rollout_kwargs,
            )

        # ground truth excludes t0
        ground_truth = x[:, 1:1 + self.num_rollout_timesteps]

        # concatenate prediction with ground truth (along height dimension)
        # (batch_size * num_points, num_rollout_timesteps, num_channels) ->
        # (2 * batch_size * num_points, num_rollout_timesteps, num_channels)
        trajectories = torch.concat([ground_truth, predictions])

        # free memory
        del ground_truth
        del predictions

        # denormalize (from mean=0 std=1 to original value range)
        trajectories = self.dataset.denormalize(trajectories, inplace=True, dim=2)

        # calculate denormalized delta
        denormed_ground_truth, denormed_predictions = trajectories.chunk(2)
        denormalized_deltas = (denormed_ground_truth - denormed_predictions).abs()

        # concatenate prediction with delta
        # (2 * num_points, num_rollout_timesteps, num_channels) ->
        # (3 * num_points, num_rollout_timesteps, num_channels)
        trajectories = torch.concat([trajectories, denormalized_deltas])

        # get positions of points
        if "mesh_pos" in data:
            pos = data["mesh_pos"]
        elif "pos" in data:
            pos = data["pos"]
        else:
            raise NotImplementedError

        # generate gifs
        for i, trajectory in enumerate([trajectories]):
            prefix = f"{self.dataset_key}_{self.update_counter.cur_checkpoint}"
            if len(self.rollout_kwargs) > 0:
                prefix = f"{prefix}_{dict_to_string(self.rollout_kwargs, item_seperator='-')}"
            prefix = f"{prefix}_idx{idx[i]:04d}"

            # calculate velocity magnitude
            velocity = trajectory[:, :, 1:]
            velocity_magnitude = torch.sqrt(torch.sum(velocity ** 2, dim=2))

            # generate images
            self.logger.info(f"generating vmag images")
            # data has shape (3 * height, width) -> normalize each sub-image seperatle
            velocity_magnitude = einops.rearrange(
                velocity_magnitude,
                "(three num_points) num_rollout_timesteps -> num_rollout_timesteps three num_points",
                three=3,
            )
            imgs = [
                self._tensor_to_pil_torch(
                    velocity_magnitude[j],
                    progress= j / max(1, (len(velocity_magnitude) - 1)),
                    pos=pos,
                )
                for j in range(len(velocity_magnitude))
            ]
            uri = self.out / f"vmag_{prefix}.gif"
            self.logger.info(f"generating vmag gif '{uri.as_posix()}'")
            imgs[0].save(
                fp=uri,
                format="GIF",
                append_images=imgs[1:],
                save_all=True,
                duration=100,
                loop=0,
            )

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, trainer_model, batch_size, data_iter, **_):
        self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer, trainer_model=trainer_model),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )
