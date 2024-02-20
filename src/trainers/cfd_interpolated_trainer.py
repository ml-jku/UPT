from functools import cached_property

import einops
from kappadata.wrappers import ModeWrapper
from torch import nn

from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback
from datasets.collators.cfd_interpolated_collator import CfdInterpolatedCollator
from losses import loss_fn_from_kwargs
from utils.factory import create
from .base.sgd_trainer import SgdTrainer


class CfdInterpolatedTrainer(SgdTrainer):
    def __init__(self, loss_function, max_batch_size=None, **kwargs):
        # automatic batchsize is not supported with mesh data
        disable_gradient_accumulation = max_batch_size is None
        super().__init__(
            max_batch_size=max_batch_size,
            disable_gradient_accumulation=disable_gradient_accumulation,
            **kwargs,
        )
        self.loss_function = create(loss_function, loss_fn_from_kwargs, update_counter=self.update_counter)

    @cached_property
    def input_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="x")
        assert dataset.root_dataset.num_query_points is not None
        assert isinstance(collator.collator, CfdInterpolatedCollator)
        input_shape = dataset.getshape_x()
        self.logger.info(f"input_shape: {input_shape}")
        return input_shape

    @cached_property
    def output_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="target")
        assert isinstance(collator.collator, CfdInterpolatedCollator)
        output_shape = dataset.getshape_target()
        self.logger.info(f"output_shape: {output_shape}")
        return output_shape

    @cached_property
    def dataset_mode(self):
        return "interpolated query_pos timestep velocity target"

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
                x=self.to_device(item="interpolated", batch=batch, dataset_mode=dataset_mode),
                query_pos=self.to_device(item="query_pos", batch=batch, dataset_mode=dataset_mode),
                timestep=self.to_device(item="timestep", batch=batch, dataset_mode=dataset_mode),
                velocity=self.to_device(item="velocity", batch=batch, dataset_mode=dataset_mode),
                target=self.to_device(item="target", batch=batch, dataset_mode=dataset_mode),
            )
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

            return dict(total=losses["x_hat"], **losses), {}
