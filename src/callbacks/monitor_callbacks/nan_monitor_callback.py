import torch

from callbacks.base.callback_base import CallbackBase


class NanMonitorCallback(CallbackBase):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose

    def _before_training(self, model, **kwargs):
        for name, module in model.named_modules():
            module.register_forward_hook(self.NanMonitorHook(name, logger=self.logger))


    class NanMonitorHook:
        def __init__(self, name: str, logger):
            self.name = name
            self.logger = logger

        def _log_and_print(self, msg):
            self.logger.info(msg)
            print(msg)

        def _log(self, module, module_input, module_output, nan_tensor_name):
            self.logger.error(f"encountered nan in module {self.name} ({nan_tensor_name})")
            self.logger.info(f"parameters:")
            for name, param in module.named_parameters():
                self._log_and_print(f"{name}.abs().max(): {param.abs().max().item()}")
                self._log_and_print(f"{name}.abs().min(): {param.abs().min().item()}")
                self._log_and_print(f"{name}.mean(): {param.mean().item()}")
                self._log_and_print(f"{name}.std(): {param.std().item()}")

            for i in range(len(module_input)):
                tensor = module_input[i].flatten()
                tensor = tensor[~torch.isnan(tensor)]
                name = f"module_input[{i}]"
                self._log_and_print(f"{name}.abs().max(): {tensor.abs().max().item()}")
                self._log_and_print(f"{name}.abs().min(): {tensor.abs().min().item()}")
                self._log_and_print(f"{name}.mean(): {tensor.mean().item()}")
                self._log_and_print(f"{name}.std(): {tensor.std().item()}")

            for i in range(len(module_output)):
                tensor = module_output[i].flatten()
                tensor = tensor[~torch.isnan(tensor)]
                name = f"module_output[{i}]"
                self._log_and_print(f"{name}.abs().max(): {tensor.abs().max().item()}")
                self._log_and_print(f"{name}.abs().min(): {tensor.abs().min().item()}")
                self._log_and_print(f"{name}.mean(): {tensor.mean().item()}")
                self._log_and_print(f"{name}.std(): {tensor.std().item()}")

            self.logger.error(f"encountered nan in module {self.name} ({nan_tensor_name})")
            exit(0)

        def __call__(self, module, module_input, module_output):
            assert isinstance(module_input, tuple)
            for i in range(len(module_input)):
                if torch.is_tensor(module_input[i]) and torch.any(torch.isnan(module_input[i])):
                    self._log(
                        module=module,
                        module_input=module_input,
                        module_output=module_output,
                        nan_tensor_name=f"module_input[{i}]"
                    )

            if isinstance(module_output, tuple):
                for i in range(len(module_output)):
                    if module_output[i] is None:
                        continue
                    assert torch.is_tensor(module_output[i])
                    if torch.any(torch.isnan(module_output[i])):
                        self._log(
                            module=module,
                            module_input=module_input,
                            module_output=module_output,
                            nan_tensor_name=f"module_output[{i}]"
                        )
