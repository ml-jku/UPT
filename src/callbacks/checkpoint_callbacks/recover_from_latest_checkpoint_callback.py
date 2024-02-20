from collections import defaultdict

import torch

from callbacks.base.periodic_callback import PeriodicCallback
from distributed.config import is_rank0
from utils.select_with_path import select_with_path
from initializers.resume_initializer import ResumeInitializer
from callbacks.checkpoint_callbacks.checkpoint_callback import CheckpointCallback

class RecoverFromLatestCheckpointCallback(PeriodicCallback):
    def __init__(self, save_discarded_checkpoints, max_num_recoveries=5, **kwargs):
        super().__init__(**kwargs)
        self.save_discarded_checkpoints = save_discarded_checkpoints
        self.max_num_recoveries = max_num_recoveries
        self.num_recoveries = 0
        self.best_loss = float("inf")
        self.latest_checkpoint_callback = None

    def _before_training(self, trainer, **kwargs):
        # TODO should handle saving latest checkpoint on its own instead of relying on other callback
        # TODO this would allow e.g. looping 2 epochs back
        assert self.every_n_epochs == trainer.log_every_n_epochs
        assert self.every_n_updates == trainer.log_every_n_updates
        assert self.every_n_samples == trainer.log_every_n_samples
        # get indices of RecoverFromLatestCheckpointCallback and CheckpointCallbacks that log the latest checkpoint
        recover_idxs = []
        ckpt_idxs = []
        for i, callback in enumerate(trainer.callbacks):
            if isinstance(callback, RecoverFromLatestCheckpointCallback):
                recover_idxs.append(i)
            if (
                    isinstance(callback, CheckpointCallback)
                    and callback.save_latest_weights
                    and callback.save_latest_optim
            ):
                ckpt_idxs.append(i)
        # make sure there is only 1 of each callback
        assert len(recover_idxs) == 1
        assert len(ckpt_idxs) == 1
        # checkpoint callback has to be after restart callback
        assert recover_idxs[0] < ckpt_idxs[0]
        # remember CheckpointCallback to disable writing after a resume
        self.latest_checkpoint_callback = trainer.callbacks[ckpt_idxs[0]]

    def state_dict(self):
        return dict(
            num_recoveries=self.num_recoveries,
            best_loss=self.best_loss,
        )

    def load_state_dict(self, state_dict):
        self.num_recoveries = state_dict["num_recoveries"]
        self.best_loss = state_dict["best_loss"]

    def _periodic_callback(self, model, **kwargs):
        # extract loss from log_cache (produced by OnlineLossCallback)
        loss = self.writer.log_cache[f"loss/online/total/{self.to_short_interval_string()}"]
        if loss > 2 * self.best_loss:
            if self.max_num_recoveries is not None and self.num_recoveries >= self.max_num_recoveries:
                raise RuntimeError("maximum number of recoveries reached ({self.max_num_recoveries})")
            self.num_recoveries += 1
            self.logger.warning(
                f"loss is higher than 2 * best loss ({loss:.6f} > 2 * {self.best_loss}) "
                f"-> recover from latest checkpoint (num_recoveries: {self.num_recoveries})"
            )
            # save current state
            if self.save_discarded_checkpoints:
                self.checkpoint_writer.save(
                    model=model,
                    checkpoint=f"faulty{self.resume_count}",
                    save_weights=True,
                    save_optim=True,
                    save_frozen_weights=True,
                )

            # recover from latest checkpoint
            initializer = ResumeInitializer(
                stage_id=self.path_provider.stage_id,
                checkpoint="latest",
                load_optim=True,
                load_random_states=False,
                path_provider=self.path_provider,
            )
            initializer.init_weights(model)
            initializer.init_optim(model)
            # prevent checkpoint callback from overwriting the last checkpoint with the current one
            if self.every_n_epochs is not None:
                self.latest_checkpoint_callback.every_n_epochs = None
            else:
                raise NotImplementedError
        else:
            # update best_loss
            if loss < self.best_loss:
                self.best_loss = loss
            # allow checkpoint callback to write new best checkpoint
            if self.every_n_epochs is not None:
                self.latest_checkpoint_callback.every_n_epochs = self.every_n_epochs

