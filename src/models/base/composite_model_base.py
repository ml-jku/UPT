import torch

from .model_base import ModelBase
from .single_model_base import SingleModelBase


class CompositeModelBase(ModelBase):
    def forward(self, *args, **kwargs):
        """ all computations for training have to be within the forward method (otherwise DDP doesn't sync grads) """
        raise NotImplementedError

    @property
    def submodels(self):
        raise NotImplementedError

    def optim_step(self, grad_scaler):
        for submodel in self.submodels.values():
            if isinstance(submodel, SingleModelBase) and submodel.optim is None:
                continue
            submodel.optim_step(grad_scaler)
        # after step (e.g. for EMA)
        self.after_update_step()

    def optim_schedule_step(self):
        for submodel in self.submodels.values():
            if isinstance(submodel, SingleModelBase) and submodel.optim is None:
                continue
            submodel.optim_schedule_step()

    def optim_zero_grad(self, set_to_none=True):
        for submodel in self.submodels.values():
            if isinstance(submodel, SingleModelBase) and submodel.optim is None:
                continue
            submodel.optim_zero_grad(set_to_none)

    @property
    def is_frozen(self):
        return all(m.is_frozen for m in self.submodels.values())

    @is_frozen.setter
    def is_frozen(self, value):
        for m in self.submodels.values():
            m.is_frozen = value

    @property
    def device(self):
        devices = [sub_model.device for sub_model in self.submodels.values()]
        assert all(device == devices[0] for device in devices[1:])
        return devices[0]

    def clear_buffers(self):
        for submodel in self.submodels.values():
            submodel.clear_buffers()

    @property
    def is_batch_size_dependent(self):
        return any(m.is_batch_size_dependent for m in self.submodels.values())

    def initialize_weights(self):
        for sub_model in self.submodels.values():
            sub_model.initialize_weights()
        if self.model_specific_initialization != ModelBase.model_specific_initialization:
            self.logger.info(f"applying model specific initialization")
            self.model_specific_initialization()
        else:
            self.logger(f"no model specific initialization")
        return self

    def apply_initializers(self):
        for sub_model in self.submodels.values():
            sub_model.apply_initializers()
        for initializer in self.initializers:
            initializer.init_weights(self)
            initializer.init_optim(self)
        return self

    def initialize_optim(self, lr_scale_factor=None):
        for submodel in self.submodels.values():
            submodel.initialize_optim(lr_scale_factor=lr_scale_factor)
        if self.is_frozen:
            self.logger.info(f"{self.name} has only frozen submodels -> put into eval mode")
            self.eval()

    def train(self, mode=True):
        for sub_model in self.submodels.values():
            sub_model.train(mode=mode)
        # avoid setting mode to train if whole network is frozen
        if self.is_frozen and mode is True:
            return
        return super().train(mode=mode)

    def to(self, device, *args, **kwargs):
        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        for sub_model in self.submodels.values():
            sub_model.to(*args, **kwargs, device=device)
        return super().to(*args, **kwargs, device=device)
