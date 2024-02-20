import torch
from functools import partial

from callbacks.base.periodic_callback import PeriodicCallback
from utils.object_from_kwargs import objects_from_kwargs


class OfflinePredCallback(PeriodicCallback):
    def __init__(self, dataset_key, forward_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.forward_kwargs = objects_from_kwargs(forward_kwargs)
        self.__config_id = None
        self.out = None

    def _register_sampler_configs(self, trainer):
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode=trainer.dataset_mode)
        self.out = self.path_provider.stage_output_path / "pred"
        self.out.mkdir(exist_ok=True)

    @staticmethod
    def _forward(batch, trainer_model, trainer, model):
        data = trainer_model.prepare(batch)
        target = data.pop("target")
        with trainer.autocast_context:
            outputs = model(**data)
        return outputs["x_hat"], target.clone()

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer_model, trainer, batch_size, data_iter, **_):
        x_hat, target = self.iterate_over_dataset(
            forward_fn=partial(self._forward, trainer_model=trainer_model, trainer=trainer, model=model),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )

        x_hat_uri = self.out / f"pred_{self.dataset_key}_{self.update_counter.cur_checkpoint}.th"
        self.logger.info(f"saving predictions to: {x_hat_uri.as_posix()}")
        torch.save(x_hat, x_hat_uri)
        torch.save(target, self.out / f"target_{self.dataset_key}_{self.update_counter.cur_checkpoint}.th")
