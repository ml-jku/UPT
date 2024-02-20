from models.extractors.finalizers import finalizer_from_kwargs
from models.extractors.finalizers.concat_finalizer import ConcatFinalizer
from utils.factory import create
from utils.select_with_path import select_with_path


class ExtractorBase:
    def __init__(
            self,
            pooling=None,
            raise_exception=False,
            finalizer=ConcatFinalizer,
            model_path=None,
            hook_kwargs=None,
            outputs=None,
            static_ctx=None,
            add_model_path_to_repr=True,
    ):
        self.pooling = pooling
        self.raise_exception = raise_exception
        # "self.outputs = outputs or {}" does not work here as an empty dictionary evaluates to false
        if outputs is None:
            self.outputs = {}
        else:
            self.outputs = outputs
        self.hooks = []
        self.finalizer = create(finalizer, finalizer_from_kwargs)
        self.model_path = model_path
        self.static_ctx = static_ctx
        self.registered_hooks = False
        self.hook_kwargs = hook_kwargs or {}
        # model paths cant contain a . if the extractor is registered as part of a module
        self.add_model_path_to_repr = add_model_path_to_repr

    def __enter__(self):
        self.enable_hooks()

    def __exit__(self, *_, **__):
        self.disable_hooks()

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.add_model_path_to_repr and self.model_path is not None:
            model_path = f"{self.model_path}."
        else:
            model_path = ""
        finalize_str = f".{str(self.finalizer)}" if not isinstance(self.finalizer, ConcatFinalizer) else ""
        return f"{model_path}{self.to_string()}{finalize_str}"

    def to_string(self):
        raise NotImplementedError

    def register_hooks(self, model):
        assert not self.registered_hooks
        model = select_with_path(obj=model, path=self.model_path)
        assert model is not None, f"model.{self.model_path} is None"
        self._register_hooks(model)
        self.registered_hooks = True
        return self

    def _register_hooks(self, model):
        raise NotImplementedError

    def enable_hooks(self, raise_exception=None):
        for hook in self.hooks:
            hook.enabled = True
            if raise_exception is not None:
                hook.raise_exception = raise_exception

    def disable_hooks(self):
        for hook in self.hooks:
            hook.enabled = False

    def _get_own_outputs(self):
        raise NotImplementedError

    def extract(self, finalizer=None, clear_outputs=True):
        assert len(self.outputs) > 0, f"no outputs for {self}"
        features = [
            self.pooling(output, ctx=self.static_ctx)
            if self.pooling is not None else
            output
            for output in self._get_own_outputs().values()
        ]
        if finalizer is not None:
            if finalizer == "none":
                pass
            else:
                raise NotImplementedError
        elif self.finalizer is not None:
            features = self.finalizer(features)
        if clear_outputs:
            self.outputs.clear()
        return features
