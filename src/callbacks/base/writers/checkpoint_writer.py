import logging

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel

from distributed.config import is_rank0
from models.base.composite_model_base import CompositeModelBase
from models.base.single_model_base import SingleModelBase
from providers.path_provider import PathProvider
from utils.checkpoint import Checkpoint
from utils.update_counter import UpdateCounter


class CheckpointWriter:
    def __init__(self, path_provider: PathProvider, update_counter: UpdateCounter):
        self.logger = logging.getLogger(type(self).__name__)
        self.path_provider = path_provider
        self.update_counter = update_counter

    def _to_ckpt_dict(self, model, ckpt):
        if isinstance(ckpt, Checkpoint):
            ckpt = dict(ckpt)
        return dict(
            state_dict=model.state_dict(),
            ctor_kwargs=model.ctor_kwargs,
            ckpt=ckpt,
            abs_ckpt=dict(self.update_counter.cur_checkpoint),
            stage_id=self.path_provider.stage_id,
        )

    def save(
            self,
            model,
            checkpoint,
            trainer=None,
            save_weights=True,
            save_optim=True,
            save_latest_weights=False,
            save_latest_optim=False,
            model_name_to_save=None,
            save_frozen_weights=False,
    ):
        # NOTE: this has to be called from all ranks because random states are gathered to rank0
        trainer_sd = trainer.state_dict() if trainer is not None else None
        if is_rank0():
            self._save_seperate_models(
                name=model.name,
                model=model,
                ckpt=checkpoint,
                save_weights=save_weights,
                save_optim=save_optim,
                save_latest_weights=save_latest_weights,
                save_latest_optim=save_latest_optim,
                model_name_to_save=model_name_to_save,
                save_frozen_weights=save_frozen_weights,
            )
            if trainer_sd is not None:
                if save_weights or save_optim:
                    trainer_out_path = self.path_provider.checkpoint_path / f"trainer cp={checkpoint}.th"
                    torch.save(trainer_sd, trainer_out_path)
                    self.logger.info(f"saved trainer state_dict to {trainer_out_path}")
                if save_latest_weights or save_latest_optim:
                    latest_trainer_out_path = self.path_provider.checkpoint_path / f"trainer cp=latest.th"
                    torch.save(trainer_sd, latest_trainer_out_path)
                    self.logger.info(f"saved trainer state_dict to {latest_trainer_out_path}")

    def _save_seperate_models(
            self,
            name,
            model,
            ckpt,
            save_weights,
            save_optim,
            save_latest_weights,
            save_latest_optim,
            model_name_to_save,
            save_frozen_weights,
    ):
        assert not isinstance(model, DistributedDataParallel)
        if isinstance(model, SingleModelBase):
            if model.is_frozen and not save_frozen_weights:
                return
            if model_name_to_save is not None and name != model_name_to_save:
                return
            # save weights with ctor_kwargs
            if save_weights:
                model_uri = self.path_provider.checkpoint_path / f"{name} cp={ckpt} model.th"
                torch.save(self._to_ckpt_dict(model=model, ckpt=ckpt), model_uri)
                self.logger.info(f"saved {name} to {model_uri}")
            if save_latest_weights:
                # save only latest weights (and overwrite old latest weights)
                model_uri = self.path_provider.checkpoint_path / f"{name} cp=latest model.th"
                torch.save(self._to_ckpt_dict(model=model, ckpt=ckpt), model_uri)
                self.logger.info(f"saved {name} to {model_uri}")
            # save optim
            if model.optim is not None:
                if save_optim:
                    optim_uri = self.path_provider.checkpoint_path / f"{name} cp={ckpt} optim.th"
                    torch.save(model.optim.state_dict(), optim_uri)
                    self.logger.info(f"saved {name} optim to {optim_uri}")
                if save_latest_optim:
                    # save only latest optim (and overwrite old latest optim)
                    optim_uri = self.path_provider.checkpoint_path / f"{name} cp=latest optim.th"
                    torch.save(model.optim.state_dict(), optim_uri)
                    self.logger.info(f"saved {name} optim to {optim_uri}")

            # save ctor kwargs
            # if save_weights:
            #     kwargs_uri = self.path_provider.checkpoint_path / f"{name} kwargs.yaml"
            #     if not kwargs_uri.exists():
            #         with open(kwargs_uri, "w") as f:
            #             yaml.safe_dump(model.ctor_kwargs, f)
        elif isinstance(model, CompositeModelBase):
            for k, v in model.submodels.items():
                self._save_seperate_models(
                    name=f"{name}.{k}",
                    model=v,
                    ckpt=ckpt,
                    save_weights=save_weights,
                    save_optim=save_optim,
                    save_latest_weights=save_latest_weights,
                    save_latest_optim=save_latest_optim,
                    model_name_to_save=model_name_to_save,
                    save_frozen_weights=save_frozen_weights,
                )
            # save ctor kwargs
            if model_name_to_save is not None:
                kwargs_uri = self.path_provider.checkpoint_path / f"{name} kwargs.yaml"
                if not kwargs_uri.exists():
                    with open(kwargs_uri, "w") as f:
                        yaml.safe_dump(model.ctor_kwargs, f)
        else:
            raise NotImplementedError
