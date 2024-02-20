from callbacks.base.periodic_callback import PeriodicCallback
from models.base.composite_model_base import CompositeModelBase
from optimizers.interleaved_optimizer import InterleavedOptimizer
from utils.model_utils import get_named_models


class LrCallback(PeriodicCallback):
    def should_log_after_update(self, checkpoint):
        if checkpoint.update == 1:
            return True
        return super().should_log_after_update(checkpoint)

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, **_):
        for model_name, model in get_named_models(model).items():
            if isinstance(model, CompositeModelBase) or model.optim is None:
                continue
            if isinstance(model.optim, InterleavedOptimizer):
                optim = model.optim.get_optim_for_previous_step()
            else:
                optim = model.optim
            for param_group in optim.torch_optim.param_groups:
                group_name = f"/{param_group['name']}" if "name" in param_group else ""
                if optim.schedule is not None:
                    lr = param_group["lr"]
                    self.writer.add_scalar(f"optim/lr/{model_name}{group_name}", lr)
                if optim.weight_decay_schedule is not None:
                    wd = param_group["weight_decay"]
                    self.writer.add_scalar(f"optim/wd/{model_name}{group_name}", wd)
