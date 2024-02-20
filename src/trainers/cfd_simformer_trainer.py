from functools import cached_property

import kappamodules.utils.tensor_cache as tc
import torch
from kappadata.wrappers import ModeWrapper
from torch import nn
from torch_geometric.nn.pool import radius_graph
from torch_scatter import segment_csr

from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback
from datasets.collators.cfd_simformer_collator import CfdSimformerCollator
from losses import loss_fn_from_kwargs
from utils.checkpoint import Checkpoint
from utils.factory import create
from .base.sgd_trainer import SgdTrainer


class CfdSimformerTrainer(SgdTrainer):
    def __init__(
            self,
            loss_function,
            detach_reconstructions=False,
            reconstruct_from_target=False,
            reconstruct_prev_x_weight=0,
            reconstruct_dynamics_weight=0,
            radius_graph_r=None,
            radius_graph_max_num_neighbors=None,
            max_batch_size=None,
            mask_loss_start_checkpoint=None,
            mask_loss_threshold=None,
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
        self.detach_reconstructions = detach_reconstructions
        self.reconstruct_from_target = reconstruct_from_target
        self.reconstruct_prev_x_weight = reconstruct_prev_x_weight
        self.reconstruct_dynamics_weight = reconstruct_dynamics_weight
        self.radius_graph_r = radius_graph_r
        self.radius_graph_max_num_neighbors = radius_graph_max_num_neighbors
        self.mask_loss_start_checkpoint = create(mask_loss_start_checkpoint, Checkpoint)
        if self.mask_loss_start_checkpoint is not None:
            assert self.mask_loss_start_checkpoint.is_minimally_specified
            self.mask_loss_start_checkpoint = self.mask_loss_start_checkpoint.to_fully_specified(
                updates_per_epoch=self.update_counter.updates_per_epoch,
                effective_batch_size=self.update_counter.effective_batch_size,
            )
        self.mask_loss_threshold = mask_loss_threshold
        self.num_supernodes = None

    def get_trainer_callbacks(self, model=None):
        keys = ["degree/input"]
        patterns = ["loss_stats", "tensor_stats"]
        return [
            UpdateOutputCallback(
                keys=keys,
                patterns=patterns,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
            UpdateOutputCallback(
                keys=keys,
                patterns=patterns,
                **self.get_default_callback_intervals(),
                **self.get_default_callback_kwargs(),
            ),
        ]

    @cached_property
    def input_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="x")
        assert isinstance(collator.collator, CfdSimformerCollator)
        self.num_supernodes = collator.collator.num_supernodes
        input_shape = dataset.getshape_x()
        self.logger.info(f"input_shape: {input_shape}")
        if self.reconstruct_prev_x_weight > 0 or self.reconstruct_dynamics_weight > 0:
            # make sure query is coupled with input
            assert dataset.couple_query_with_input
        else:
            if self.end_checkpoint.is_zero:
                # eval run -> doesnt matter
                pass
            else:
                # check that num_query_points is used if no latent rollout losses are used
                # there is no reason not to use it without reconstruction losses
                assert dataset.root_dataset.num_query_points is not None
        return input_shape

    @cached_property
    def output_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="x")
        assert isinstance(collator.collator, CfdSimformerCollator)
        output_shape = dataset.getshape_target()
        self.logger.info(f"output_shape: {output_shape}")
        return output_shape

    @cached_property
    def dataset_mode(self):
        return "x mesh_pos query_pos mesh_edges geometry2d timestep velocity target"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(nn.Module):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer
            # self.counter = 0

        def to_device(self, item, batch, dataset_mode):
            data = ModeWrapper.get_item(mode=dataset_mode, item=item, batch=batch)
            data = data.to(self.model.device, non_blocking=True)
            return data

        def prepare(self, batch, dataset_mode=None):
            dataset_mode = dataset_mode or self.trainer.dataset_mode
            batch, ctx = batch
            mesh_pos = self.to_device(item="mesh_pos", batch=batch, dataset_mode=dataset_mode)
            batch_idx = ctx["batch_idx"].to(self.model.device, non_blocking=True)
            data = dict(
                x=self.to_device(item="x", batch=batch, dataset_mode=dataset_mode),
                geometry2d=self.to_device(item="geometry2d", batch=batch, dataset_mode=dataset_mode),
                timestep=self.to_device(item="timestep", batch=batch, dataset_mode=dataset_mode),
                velocity=self.to_device(item="velocity", batch=batch, dataset_mode=dataset_mode),
                query_pos=self.to_device(item="query_pos", batch=batch, dataset_mode=dataset_mode),
                mesh_pos=mesh_pos,
                batch_idx=batch_idx,
                unbatch_idx=ctx["unbatch_idx"].to(self.model.device, non_blocking=True),
                unbatch_select=ctx["unbatch_select"].to(self.model.device, non_blocking=True),
                target=self.to_device(item="target", batch=batch, dataset_mode=dataset_mode),
            )
            mesh_edges = ModeWrapper.get_item(item="mesh_edges", batch=batch, mode=dataset_mode)
            if mesh_edges is None:
                # create mesh edges on GPU
                assert self.trainer.radius_graph_r is not None
                assert self.trainer.radius_graph_max_num_neighbors is not None
                if self.trainer.num_supernodes is None:
                    # normal flow direction
                    flow = "source_to_target"
                    supernode_idxs = None
                else:
                    # inverted flow direction is required to have sorted dst_indices
                    flow = "target_to_source"
                    supernode_idxs = ctx["supernode_idxs"].to(self.model.device, non_blocking=True)
                mesh_edges = radius_graph(
                    x=mesh_pos,
                    r=self.trainer.radius_graph_r,
                    max_num_neighbors=self.trainer.radius_graph_max_num_neighbors,
                    batch=batch_idx,
                    loop=True,
                    flow=flow,
                )
                if supernode_idxs is not None:
                    is_supernode_edge = torch.isin(mesh_edges[0], supernode_idxs)
                    mesh_edges = mesh_edges[:, is_supernode_edge]
                mesh_edges = mesh_edges.T
            else:
                assert self.trainer.radius_graph_r is None
                assert self.trainer.radius_graph_max_num_neighbors is None
                assert self.trainer.num_supernodes is None
                mesh_edges = mesh_edges.to(self.model.device, non_blocking=True)
            data["mesh_edges"] = mesh_edges
            return data

        def forward(self, batch, reduction="mean"):
            data = self.prepare(batch=batch)

            x = data.pop("x")
            target = data.pop("target")
            batch_idx = data["batch_idx"]
            batch_size = batch_idx.max() + 1

            # forward pass
            forward_kwargs = {}
            if self.trainer.reconstruct_from_target:
                forward_kwargs["target"] = target
            model_outputs = self.model(
                x,
                **data,
                **forward_kwargs,
                detach_reconstructions=self.trainer.detach_reconstructions,
                reconstruct_prev_x=self.trainer.reconstruct_prev_x_weight > 0,
                reconstruct_dynamics=self.trainer.reconstruct_dynamics_weight > 0,
            )

            infos = {}
            losses = {}

            # next timestep loss
            x_hat_loss = self.trainer.loss_function(
                prediction=model_outputs["x_hat"],
                target=target,
                reduction="none",
            )
            infos.update(
                {
                    "loss_stats/x_hat/min": x_hat_loss.min(),
                    "loss_stats/x_hat/max": x_hat_loss.max(),
                    "loss_stats/x_hat/gt1": (x_hat_loss > 1).sum() / x_hat_loss.numel(),
                    "loss_stats/x_hat/eq0": (x_hat_loss == 0).sum() / x_hat_loss.numel(),
                }
            )
            # mask high values after some time to avoid instabilities
            if self.trainer.mask_loss_start_checkpoint is not None:
                if self.trainer.mask_loss_start_checkpoint > self.trainer.update_counter.cur_checkpoint:
                    x_hat_loss_mask = x_hat_loss > self.trainer.mask_loss_threshold
                    x_hat_loss = x_hat_loss[x_hat_loss_mask]
                    infos["loss_stats/x_hat/gt_loss_threshold"] = x_hat_loss_mask.sum() / x_hat_loss_mask.numel()
            if reduction == "mean":
                losses["x_hat"] = x_hat_loss.mean()
            elif reduction == "mean_per_sample":
                _, ctx = batch
                num_zero_pos = (data["query_pos"] == 0).sum()
                assert num_zero_pos == 0, f"padded query_pos not supported {num_zero_pos}"
                query_pos_len = data["query_pos"].size(1)
                query_batch_idx = torch.arange(batch_size, device=self.model.device).repeat_interleave(query_pos_len)
                #query_batch_idx = ctx["query_batch_idx"].to(self.model.device, non_blocking=True)
                # indptr is a tensor of indices betweeen which to aggregate
                # i.e. a tensor of [0, 2, 5] would result in [src[0] + src[1], src[2] + src[3] + src[4]]
                indices, counts = query_batch_idx.unique(return_counts=True)
                # first index has to be 0
                padded_counts = torch.zeros(len(indices) + 1, device=counts.device, dtype=counts.dtype)
                padded_counts[indices + 1] = counts
                indptr = padded_counts.cumsum(dim=0)
                losses["x_hat"] = segment_csr(src=x_hat_loss.mean(dim=1), indptr=indptr, reduce="mean")
            else:
                raise NotImplementedError
            total_loss = losses["x_hat"]


            # num_objects = self.to_device(item="num_objects", batch=batch[0], dataset_mode=self.trainer.dataset_mode),
            # out = self.trainer.path_provider.stage_output_path / f"tensors"
            # out.mkdir(exist_ok=True)
            # torch.save(num_objects, out / f"{self.counter:04d}_numobjects.th")
            # torch.save(x, out / f"{self.counter:04d}_x.th")
            # torch.save(target, out / f"{self.counter:04d}_target.th")
            # torch.save(data["timestep"], out / f"{self.counter:04d}_timestep.th")
            # torch.save(data["velocity"], out / f"{self.counter:04d}_velocity.th")
            # torch.save(data["query_pos"], out / f"{self.counter:04d}_querypos.th")
            # torch.save(data["mesh_pos"], out / f"{self.counter:04d}_meshpos.th")
            # torch.save(data["batch_idx"], out / f"{self.counter:04d}_batchidx.th")
            # torch.save(data["mesh_edges"], out / f"{self.counter:04d}_meshedges.th")
            # torch.save(model_outputs["x_hat"], out / f"{self.counter:04d}_xhat.th")
            # self.counter += 1

            # input_reconstruction losses
            if self.trainer.reconstruct_prev_x_weight > 0:
                num_channels = model_outputs["prev_x_hat"].size(1)
                prev_x_hat_loss = self.trainer.loss_function(
                    prediction=model_outputs["prev_x_hat"],
                    target=x[:, -num_channels:],
                    reduction="none",
                )
                if reduction == "mean":
                    # mask out reconstruction for timestep==0
                    timestep = data["timestep"]
                    timestep_per_point = torch.gather(timestep, dim=0, index=batch_idx)
                    prev_x_hat_loss = prev_x_hat_loss[timestep_per_point != 0]
                    if self.trainer.mask_loss_start_checkpoint is not None:
                        if self.trainer.mask_loss_start_checkpoint > self.trainer.update_counter.cur_checkpoint:
                            prev_x_hat_loss_mask = prev_x_hat_loss > self.trainer.mask_loss_threshold
                            prev_x_hat_loss = prev_x_hat_loss[prev_x_hat_loss_mask]
                            infos["loss_stats/prev_x_hat/gt_loss_threshold"] = \
                                prev_x_hat_loss_mask.sum() / prev_x_hat_loss_mask.numel()
                    prev_x_hat_loss = prev_x_hat_loss.mean()
                elif reduction == "mean_per_sample":
                    raise NotImplementedError
                    # prev_x_hat_loss = prev_x_hat_loss.flatten(start_dim=1).mean(dim=1)
                    # # set loss for timestep==0 to 0
                    # prev_x_hat_loss[is_timestep0] = 0.
                else:
                    raise NotImplementedError
                losses["prev_x_hat"] = prev_x_hat_loss
                total_loss = total_loss + self.trainer.reconstruct_prev_x_weight * prev_x_hat_loss

            # dynamics reconstruction losses
            if self.trainer.reconstruct_dynamics_weight > 0:
                dynamics_hat_loss = self.trainer.loss_function(
                    prediction=model_outputs["dynamics_hat"],
                    target=model_outputs["dynamics"],
                    reduction="none",
                )
                max_timestep = self.model.conditioner.num_total_timesteps - 1
                timestep = data["timestep"]
                if reduction == "mean":
                    # mask out reconstruction for timestep==T
                    dynamics_hat_mask = timestep != max_timestep
                    if dynamics_hat_mask.sum() > 0:
                        dynamics_hat_loss = dynamics_hat_loss[dynamics_hat_mask].mean()
                    else:
                        dynamics_hat_loss = tc.zeros(size=(1,), device=timestep.device)
                elif reduction == "mean_per_sample":
                    # set loss for timestep==0 to 0
                    dynamics_hat_loss[timestep == max_timestep] = 0.
                    # average per sample
                    dynamics_hat_loss = dynamics_hat_loss.flatten(start_dim=1).mean(dim=1)
                else:
                    raise NotImplementedError
                losses["dynamics_hat"] = dynamics_hat_loss
                total_loss = total_loss + self.trainer.reconstruct_dynamics_weight * dynamics_hat_loss

            infos.update(
                {
                    # "tensor_stats/x/absmax": x.abs().max(),
                    # "tensor_stats/x/absmin": x.abs().max(),
                    # "tensor_stats/x/mean": x.mean(),
                    # "tensor_stats/x/absmean": x.abs().mean(),
                    # "tensor_stats/x/std": x.std(),
                    # "tensor_stats/target/absmax": target.abs().max(),
                    # "tensor_stats/target/absmin": target.abs().max(),
                    # "tensor_stats/target/mean": target.mean(),
                    # "tensor_stats/target/absmean": target.abs().mean(),
                    # "tensor_stats/target/std": target.std(),
                    # "tensor_stats/timestep/max": data["timestep"].max(),
                    # "tensor_stats/timestep/min": data["timestep"].min(),
                    # "tensor_stats/timestep/mean": data["timestep"].float().mean(),
                    # "tensor_stats/timestep/std": data["timestep"].float().std(),
                    # "tensor_stats/velocity/max": data["velocity"].max(),
                    # "tensor_stats/velocity/min": data["velocity"].min(),
                    # "tensor_stats/velocity/mean": data["velocity"].float().mean(),
                    # "tensor_stats/velocity/std": data["velocity"].float().std(),
                    # "tensor_stats/x_hat/absmax": model_outputs["x_hat"].abs().max(),
                    # "tensor_stats/x_hat/absmin": model_outputs["x_hat"].abs().min(),
                    # "tensor_stats/x_hat/mean": model_outputs["x_hat"].mean(),
                    # "tensor_stats/x_hat/absmean": model_outputs["x_hat"].abs().mean(),
                    # "tensor_stats/x_hat/std": model_outputs["x_hat"].abs().std(),
                },
            )
            # calculate degree of graph (average number of connections p)
            # TODO: degree is incorrectly calculated if num_supernodes is handled by dataset and not by collator
            if self.trainer.num_supernodes is None:
                infos["degree/input"] = len(data["mesh_edges"]) / len(x)
            else:
                infos["degree/input"] = len(data["mesh_edges"]) / (self.trainer.num_supernodes * batch_size)

            return dict(total=total_loss, **losses), infos
