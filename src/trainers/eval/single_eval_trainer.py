import torch
import torch.nn as nn
from kappadata.wrappers import ModeWrapper

from trainers.base.sgd_trainer import SgdTrainer


class SingleEvalTrainer(SgdTrainer):
    def __init__(
            self,
            max_epochs=0,
            precision="float32",
            effective_batch_size=2,
            disable_gradient_accumulation=True,
            **kwargs,
    ):
        super().__init__(
            max_epochs=max_epochs,
            precision=precision,
            effective_batch_size=effective_batch_size,
            disable_gradient_accumulation=disable_gradient_accumulation,
            **kwargs,
        )

    @property
    def output_shape(self):
        return 2,

    @property
    def dataset_mode(self):
        return f"index x"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(nn.Module):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        def forward(self, batch):
            batch, ctx = batch
            x = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="x", batch=batch)
            x = x.to(self.model.device, non_blocking=True)
            _ = self.model(x)
            return dict(total=torch.tensor(0.)), {}
