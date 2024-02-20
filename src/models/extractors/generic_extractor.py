from torch import nn

from .base.extractor_base import ExtractorBase
from .base.forward_hook import ForwardHook


class GenericExtractor(ExtractorBase):
    def to_string(self):
        return f"GenericExtractor({self.model_path})"

    def _get_own_outputs(self):
        return {self.model_path: self.outputs[self.model_path]}

    def _register_hooks(self, model):
        hook = ForwardHook(outputs=self.outputs, output_name=self.model_path, **self.hook_kwargs)
        model.register_forward_hook(hook)
        self.hooks.append(hook)
