import torch
from functools import cached_property

import einops
from kappadata.wrappers import ModeWrapper
from torch import nn

from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback
from datasets.collators.cfd_hybrid_collator import CfdHybridCollator
from losses import loss_fn_from_kwargs
from utils.factory import create
from .base.sgd_trainer import SgdTrainer
from torch_geometric.nn.pool import radius


class CfdHybridTrainer(SgdTrainer):
    def __init__(
            self,
            loss_function,
            radius_graph_r=None,
            radius_graph_max_num_neighbors=None,
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
        self.radius_graph_r = radius_graph_r
        self.radius_graph_max_num_neighbors = radius_graph_max_num_neighbors
        self.loss_function = create(loss_function, loss_fn_from_kwargs, update_counter=self.update_counter)

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
        assert dataset.root_dataset.num_query_points is not None
        assert isinstance(collator.collator, CfdHybridCollator)
        input_shape = dataset.getshape_x()
        self.logger.info(f"input_shape: {input_shape}")
        return input_shape

    @cached_property
    def output_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="x")
        assert isinstance(collator.collator, CfdHybridCollator)
        output_shape = dataset.getshape_target()
        self.logger.info(f"output_shape: {output_shape}")
        return output_shape

    @cached_property
    def dataset_mode(self):
        return "x mesh_pos grid_pos query_pos mesh_to_grid_edges timestep velocity target"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(nn.Module):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        def to_device(self, item, batch, dataset_mode):
            data = ModeWrapper.get_item(mode=dataset_mode, item=item, batch=batch)
            data = data.to(self.model.device, non_blocking=True)
            return data

        def prepare(self, batch, dataset_mode=None):
            dataset_mode = dataset_mode or self.trainer.dataset_mode
            batch, ctx = batch
            data = dict(
                x=self.to_device(item="x", batch=batch, dataset_mode=dataset_mode),
                mesh_pos=self.to_device(item="mesh_pos", batch=batch, dataset_mode=dataset_mode),
                grid_pos=self.to_device(item="grid_pos", batch=batch, dataset_mode=dataset_mode),
                timestep=self.to_device(item="timestep", batch=batch, dataset_mode=dataset_mode),
                velocity=self.to_device(item="velocity", batch=batch, dataset_mode=dataset_mode),
                query_pos=self.to_device(item="query_pos", batch=batch, dataset_mode=dataset_mode),
                unbatch_idx=ctx["unbatch_idx"].to(self.model.device, non_blocking=True),
                unbatch_select=ctx["unbatch_select"].to(self.model.device, non_blocking=True),
                target=self.to_device(item="target", batch=batch, dataset_mode=dataset_mode),
            )
            mesh_to_grid_edges = ModeWrapper.get_item(item="mesh_to_grid_edges", batch=batch, mode=dataset_mode)
            batch_size = len(data["timestep"])
            if mesh_to_grid_edges is None:
                assert len(data["grid_pos"]) % batch_size == 0
                num_grid_points = len(data["grid_pos"]) // batch_size
                grid_batch_idx = torch.arange(batch_size, device=self.model.device).repeat_interleave(num_grid_points)
            else:
                grid_batch_idx = None
            # mesh_to_grid_edges
            if mesh_to_grid_edges is None:
                # create on GPU
                assert self.trainer.radius_graph_r is not None
                assert self.trainer.radius_graph_max_num_neighbors is not None
                mesh_to_grid_edges = radius(
                    x=data["mesh_pos"],
                    y=data["grid_pos"],
                    batch_x=ctx["batch_idx"].to(self.model.device, non_blocking=True),
                    batch_y=grid_batch_idx,
                    r=self.trainer.radius_graph_r,
                    max_num_neighbors=self.trainer.radius_graph_max_num_neighbors,
                ).T
            else:
                assert self.trainer.radius_graph_r is None
                assert self.trainer.radius_graph_max_num_neighbors is None
                mesh_to_grid_edges = mesh_to_grid_edges.to(self.model.device, non_blocking=True)
            data["mesh_to_grid_edges"] = mesh_to_grid_edges


            return data

        def forward(self, batch, reduction="mean"):
            data = self.prepare(batch)
            target = data.pop("target")

            # forward pass
            model_outputs = self.model(**data)
            losses = dict(
                x_hat=self.trainer.loss_function(
                    prediction=model_outputs["x_hat"],
                    target=target,
                    reduction=reduction,
                ),
            )

            if reduction == "mean_per_sample":
                raise NotImplementedError("reduce with query_batch_idx")

            # infos
            infos = {"degree/input": len(data["mesh_to_grid_edges"]) / len(data["grid_pos"])}

            return dict(total=losses["x_hat"], **losses), infos
