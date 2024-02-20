import os

import torch

from models.base.composite_model_base import CompositeModelBase
from models.base.single_model_base import SingleModelBase
from utils.checkpoint import Checkpoint
from .base.checkpoint_initializer import CheckpointInitializer


class ResumeInitializer(CheckpointInitializer):
    """
    initializes models/optims from a checkpoint ready for resuming training
    load_optim=True as this is usually used to resume a training run
    stage_name is provided by the trainer as it already knows the correct stage_name
    """

    def __init__(self, load_optim=True, load_random_states=True, **kwargs):
        super().__init__(load_optim=load_optim, model_name=None, **kwargs)
        self.load_random_states = load_random_states

    def init_weights(self, model):
        self._init_weights(model.name, model)

    def _init_weights(self, name, model):
        if isinstance(model, SingleModelBase):
            model_name, ckpt_uri = self._get_modelname_and_ckpturi(model=model, model_name=name, file_type="model")
            sd = torch.load(ckpt_uri, map_location=model.device)
            if "state_dict" in sd:
                sd = sd["state_dict"]
            model.load_state_dict(sd)
            self.logger.info(f"loaded weights of {model_name} from {ckpt_uri}")
        if isinstance(model, CompositeModelBase):
            for submodel_name, submodel in model.submodels.items():
                self._init_weights(name=f"{name}.{submodel_name}", model=submodel)

    def init_optim(self, model):
        self._init_optim(name=model.name, model=model)

    def _init_optim(self, name, model):
        if isinstance(model, SingleModelBase):
            if model.optim is None:
                # e.g. EMA target network doesn't have an optimizer
                self.logger.info(
                    f"skip loading optim from checkpoint '{self.checkpoint}' for {model.name} "
                    f"(optim is None)"
                )
            elif model.is_frozen:
                self.logger.info(
                    f"skip loading optim from checkpoint '{self.checkpoint}' for {model.name} "
                    f"(is_frozen)"
                )
            else:
                model_name, ckpt_uri = self._get_modelname_and_ckpturi(model=model, model_name=name, file_type="optim")
                sd = torch.load(ckpt_uri, map_location=model.device)
                model.optim.load_state_dict(sd)
                self.logger.info(f"loaded optimizer of {model_name} from {ckpt_uri}")
        if isinstance(model, CompositeModelBase):
            for submodel_name, submodel in model.submodels.items():
                self._init_optim(name=f"{name}.{submodel_name}", model=submodel)

    def _get_trainer_ckpt_file(self):
        return self._get_ckpt_uri(prefix=f"trainer cp=", suffix=".th")

    def get_start_checkpoint(self):
        if isinstance(self.checkpoint, str):
            trainer_ckpt_uri = self._get_trainer_ckpt_file()
            if trainer_ckpt_uri.exists():
                trainer_ckpt = torch.load(trainer_ckpt_uri)
                trainer_ckpt_without_rng_states = {k: v for k, v in trainer_ckpt.items() if k != "random_states"}
                self.logger.info(f"loaded checkpoint from trainer_state_dict: {trainer_ckpt_without_rng_states}")
                return Checkpoint(
                    epoch=trainer_ckpt["epoch"],
                    update=trainer_ckpt["update"],
                    sample=trainer_ckpt["sample"],
                )
            else:
                self.logger.warning("no trainer checkpoint found -> try to fetch start_checkpoint from a model ckpt")
                # try to get any model checkpoint
                ckpt_folder = self.path_provider.get_stage_checkpoint_path(
                    stage_name=self.stage_name,
                    stage_id=self.stage_id,
                )
                fnames = [
                    fname
                    for fname in sorted(os.listdir(ckpt_folder))
                    if "model" in fname and self.checkpoint in fname
                ]
                assert len(fnames) > 0, "no trainer checkpoint and no valid model found to infer start_checkpoint"
                fname = fnames[0]
                self.logger.info(
                    f"no trainer checkpoint found but start_checkpoint "
                    f"can be inferred from model checkpoint '{fname}'"
                )
                model_sd = torch.load(ckpt_folder / fname, map_location="cpu")
                abs_ckpt = model_sd["abs_ckpt"]
                assert isinstance(abs_ckpt, dict)
                return Checkpoint(**abs_ckpt)
        else:
            return Checkpoint.to_fully_specified_from_fnames(
                ckpt_folder=self.path_provider.get_stage_checkpoint_path(
                    stage_name=self.stage_name,
                    stage_id=self.stage_id,
                ),
                ckpt=self.checkpoint,
            )

    def init_trainer(self, trainer):
        ckpt_uri = self._get_trainer_ckpt_file()
        if not ckpt_uri.exists():
            self.logger.warning(f"no trainer checkpoint found -> skip trainer initialization from checkpoint")
            return
        trainer.load_state_dict(torch.load(ckpt_uri), load_random_states=self.load_random_states)
        self.logger.info(f"loaded trainer checkpoint {ckpt_uri}")
