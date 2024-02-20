from callbacks.base.periodic_callback import PeriodicCallback
from utils.factory import create_collection
from models.extractors import extractor_from_kwargs
from models.extractors.generic_extractor import GenericExtractor

class ActivationMonitorCallback(PeriodicCallback):
    def __init__(self, model_paths, **kwargs):
        super().__init__(**kwargs)
        self.extractors = [GenericExtractor(model_path=model_path) for model_path in model_paths]

    def _before_training(self, model, **kwargs):
        for extractor in self.extractors:
            extractor.register_hooks(model)

    def before_every_accumulation_step(self, model, **kwargs):
        for extractor in self.extractors:
            extractor.enable_hooks()

    def before_every_optim_step(self, **kwargs):
        for extractor in self.extractors:
            act = extractor.extract()
            if isinstance(act, tuple):
                act = act[0]
            name = extractor.model_path
            self.writer.add_scalar(f"act/{name}/norm", act.norm(p=2))
            self.writer.add_scalar(f"act/{name}/avgnorm", act.norm(p=2, dim=-1).mean())
            self.writer.add_scalar(f"act/{name}/absmax", act.abs().max())
            self.writer.add_scalar(f"act/{name}/absmin", act.abs().min())
            self.writer.add_scalar(f"act/{name}/mean", act.mean())
            if act.numel() > 1:
                self.writer.add_scalar(f"act/{name}/std", act.std())
            extractor.disable_hooks()
