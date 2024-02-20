import numpy as np
import torch.nn as nn

from .base.single_model_base import SingleModelBase


class DummyModel(SingleModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer = nn.Linear(np.prod(self.input_shape), np.prod(self.output_shape or self.input_shape))

    def forward(self, x, *_, **__):
        return self.layer(x.flatten(start_dim=1)).reshape(len(x), *(self.output_shape or self.input_shape))

    def predict(self, x):
        return dict(main=self(x))

    def predict_binary(self, x):
        return dict(main=self(x))

    def load_state_dict(self, state_dict, strict=True):
        pass