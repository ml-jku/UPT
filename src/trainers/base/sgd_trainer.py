import logging
from functools import partial

import kappaprofiler as kp
import torch
import torch.nn as nn
from kappadata.wrappers import KDMultiViewWrapper, XRepeatWrapper
from torch.cuda.amp import GradScaler
from torch.distributed import all_gather_object
from torch.nn.parallel import DistributedDataParallel

from callbacks import callback_from_kwargs
from callbacks.base.callback_base import CallbackBase
from callbacks.base.periodic_callback import PeriodicCallback
from callbacks.default_callbacks.copy_previous_config_callback import CopyPreviousConfigCallback
from callbacks.default_callbacks.copy_previous_summary_callback import CopyPreviousSummaryCallback
from callbacks.default_callbacks.dataset_stats_callback import DatasetStatsCallback
from callbacks.default_callbacks.eta_callback import EtaCallback
from callbacks.default_callbacks.freezer_callback import FreezerCallback
from callbacks.default_callbacks.lr_callback import LrCallback
from callbacks.default_callbacks.online_loss_callback import OnlineLossCallback
from callbacks.default_callbacks.param_count_callback import ParamCountCallback
from callbacks.default_callbacks.progress_callback import ProgressCallback
from callbacks.default_callbacks.train_time_callback import TrainTimeCallback
from distributed.config import is_distributed, get_world_size
from distributed.config import is_managed, get_rank, is_rank0
from distributed.gather import all_gather_nograd
from initializers import initializer_from_kwargs
from initializers.resume_initializer import ResumeInitializer
from providers.config_providers.base.config_provider_base import ConfigProviderBase
from providers.config_providers.noop_config_provider import NoopConfigProvider
from providers.path_provider import PathProvider
from providers.summary_providers.base.summary_provider_base import SummaryProviderBase
from providers.summary_providers.noop_summary_provider import NoopSummaryProvider
from trainers.early_stoppers import early_stopper_from_kwargs
from utils.amp_utils import get_supported_precision, get_grad_scaler_and_autocast_context
from utils.checkpoint import Checkpoint
from utils.data_container import DataContainer
from utils.factory import create
from utils.factory import create_collection
from utils.model_utils import get_paramnames_with_no_gradient
from utils.model_utils import get_trainable_param_count
from utils.seed import get_random_states
from utils.seed import set_random_states
from utils.update_counter import UpdateCounter
from .functional import (
    calculate_effective_batch_size_per_device,
    calculate_batch_size_and_accumulation_steps,
    calculate_automatic_max_batch_size,
)


class SgdTrainer(nn.Module):
    def __init__(
            self,
            data_container: DataContainer,
            device: str,
            precision,
            effective_batch_size: int = None,
            effective_labeled_batch_size: int = None,
            max_epochs=None,
            max_updates=None,
            max_samples=None,
            start_at_epoch=None,
            stop_at_epoch=None,
            stop_at_update=None,
            stop_at_sample=None,
            add_default_callbacks: bool = True,
            add_trainer_callbacks: bool = True,
            callbacks: list = None,
            backup_precision: str = None,
            log_every_n_epochs=None,
            log_every_n_updates=None,
            log_every_n_samples=None,
            track_every_n_updates=50,
            track_every_n_samples=None,
            early_stopper=None,
            exit_on_nan_loss=True,
            initializer: ResumeInitializer = None,
            disable_gradient_accumulation: bool = False,
            max_batch_size: int = None,
            sync_batchnorm: bool = True,
            # find_unused_params should not be set to true if it is not needed (to avoid overhead)
            # but sometimes it is required (e.g. when dynamically freezing/unfreezing parameters)
            # when find_unused_params setting static_graph to true can bring speedup
            find_unused_params: bool = False,
            static_graph: bool = False,
            use_torch_compile: bool = False,
            # kwargs
            main_sampler_kwargs: dict = None,
            # providers
            config_provider: ConfigProviderBase = None,
            summary_provider: SummaryProviderBase = None,
            path_provider: PathProvider = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(type(self).__name__)
        self.data_container = data_container
        self.config_provider = config_provider or NoopConfigProvider()
        self.summary_provider = summary_provider or NoopSummaryProvider()
        self.path_provider = path_provider

        self.device: torch.device = torch.device(device)
        if effective_batch_size is not None:
            assert effective_labeled_batch_size is None
            self.effective_batch_size = effective_batch_size
        else:
            assert "num_unlabeled_per_labeled" in main_sampler_kwargs
            factor = 1 + main_sampler_kwargs["num_unlabeled_per_labeled"]
            self.effective_batch_size = effective_labeled_batch_size * factor
        self.effective_labeled_batch_size = effective_labeled_batch_size
        self.end_checkpoint = Checkpoint(max_epochs, max_updates, max_samples)
        self.stop_at_epoch = stop_at_epoch
        self.stop_at_update = stop_at_update
        self.stop_at_sample = stop_at_sample
        self.add_default_callbacks = add_default_callbacks
        self.add_trainer_callbacks = add_trainer_callbacks
        self.precision = get_supported_precision(
            desired_precision=precision,
            backup_precision=backup_precision,
            device=self.device,
        )
        self.logger.info(f"using precision: {self.precision} (desired={precision} backup={backup_precision})")
        self.grad_scaler, self.autocast_context = get_grad_scaler_and_autocast_context(self.precision, self.device)
        self.log_every_n_epochs = log_every_n_epochs
        self.log_every_n_updates = log_every_n_updates
        self.log_every_n_samples = log_every_n_samples
        self.track_every_n_updates = track_every_n_updates
        self.track_every_n_samples = track_every_n_samples
        self.early_stopper = create(early_stopper, early_stopper_from_kwargs)
        self.main_sampler_kwargs = main_sampler_kwargs or {}
        self.train_dataset, self.main_collator = self.data_container.get_dataset("train", mode=self.dataset_mode)
        self.main_sampler = self.data_container.get_main_sampler(
            train_dataset=self.train_dataset,
            **self.main_sampler_kwargs,
        )
        eff_len = self.main_sampler.effective_length
        assert eff_len >= self.effective_batch_size, f"{eff_len}<{self.effective_batch_size}"
        self.updates_per_epoch = int(eff_len / self.effective_batch_size)
        self.max_batch_size = max_batch_size
        self.disable_gradient_accumulation = disable_gradient_accumulation
        self.sync_batchnorm = sync_batchnorm
        self.find_unused_params = find_unused_params
        self.static_graph = static_graph
        self.use_torch_compile = use_torch_compile
        self.exit_on_nan_loss = exit_on_nan_loss

        self.initializer = create(
            initializer,
            initializer_from_kwargs,
            path_provider=self.path_provider,
        )
        if self.initializer is None:
            if start_at_epoch is not None:
                start_epoch = start_at_epoch
                start_update = self.updates_per_epoch * start_epoch
                start_sample = start_update * effective_batch_size
            else:
                start_epoch = 0
                start_update = 0
                start_sample = 0
            self.start_checkpoint = Checkpoint(epoch=start_epoch, update=start_update, sample=start_sample)
        else:
            assert start_at_epoch is None
            self.start_checkpoint = self.initializer.get_start_checkpoint()
        self._update_counter = UpdateCounter(
            start_checkpoint=self.start_checkpoint,
            end_checkpoint=self.end_checkpoint,
            updates_per_epoch=self.updates_per_epoch,
            effective_batch_size=self.effective_batch_size,
        )
        self.callbacks = create_collection(
            callbacks,
            callback_from_kwargs,
            data_container=self.data_container,
            config_provider=self.config_provider,
            summary_provider=self.summary_provider,
            path_provider=self.path_provider,
            update_counter=self.update_counter,
        )

        # check that children only override their implementation methods
        assert type(self).train == SgdTrainer.train
        assert type(self).wrap_model == SgdTrainer.wrap_model

    @property
    def update_counter(self):
        return self._update_counter

    @property
    def input_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="x")
        sample, _ = dataset[0]
        if collator is not None:
            self.logger.warning(
                "infering input_shape with a collator is not supported yet -> "
                "collator is ignored"
            )
        multi_view_wrappers = [
            w for w in self.train_dataset.all_wrappers
            if isinstance(w, (KDMultiViewWrapper, XRepeatWrapper))
        ]
        if len(multi_view_wrappers) > 1:
            raise NotImplementedError
        elif len(multi_view_wrappers) == 1:
            input_shape = sample[0].shape
        else:
            input_shape = sample.shape
        self.logger.info(f"input_shape: {tuple(input_shape)}")
        return tuple(input_shape)

    def get_all_callbacks(self, model=None):
        # no default/trainer callbacks needed for eval runs
        if self.end_checkpoint.epoch == 0 or self.end_checkpoint.update == 0 or self.end_checkpoint.sample == 0:
            return self.callbacks

        # add default/trainer callbacks
        callbacks = []
        if self.add_default_callbacks:
            callbacks += self.get_default_callbacks()
        if self.add_trainer_callbacks:
            callbacks += self.get_trainer_callbacks(model=model)
        callbacks += self.callbacks
        return callbacks

    @staticmethod
    def get_trainer_callbacks(model=None):
        return []

    def get_default_callback_kwargs(self):
        return dict(
            data_container=self.data_container,
            config_provider=self.config_provider,
            summary_provider=self.summary_provider,
            path_provider=self.path_provider,
            update_counter=self.update_counter,
        )

    def get_default_callback_intervals(self):
        return dict(
            every_n_epochs=self.log_every_n_epochs,
            every_n_updates=self.log_every_n_updates,
            every_n_samples=self.log_every_n_samples,
        )

    def get_default_callbacks(self):
        default_kwargs = self.get_default_callback_kwargs()
        default_intervals = self.get_default_callback_intervals()
        # statistic callbacks
        default_callbacks = [
            DatasetStatsCallback(**default_kwargs),
            ParamCountCallback(**default_kwargs),
        ]
        # copy config/summary/entries
        default_callbacks += [
            CopyPreviousConfigCallback(**default_kwargs),
            # CopyPreviousEntriesCallback(**default_kwargs),
            CopyPreviousSummaryCallback(**default_kwargs),
        ]

        # add default training loggers (not needed for eval runs)
        if not self.update_counter.is_finished:
            # periodic callbacks
            default_callbacks += [
                ProgressCallback(**default_kwargs, **default_intervals),
                TrainTimeCallback(**default_kwargs, **default_intervals),
                OnlineLossCallback(**default_kwargs, **default_intervals, verbose=True),
            ]

            # EtaCallback is pointless in managed runs
            # - managed runs don't have an interactive console
            if not is_managed() and is_rank0():
                default_callbacks = [EtaCallback(**default_kwargs, **default_intervals)] + default_callbacks

            default_callbacks += [
                LrCallback(**default_kwargs, every_n_updates=self.track_every_n_updates),
                FreezerCallback(**default_kwargs, every_n_updates=self.track_every_n_updates),
                OnlineLossCallback(**default_kwargs, every_n_updates=self.track_every_n_updates, verbose=False)
            ]

        for callback in default_callbacks:
            self.logger.info(f"added default {callback}")
        return default_callbacks

    def _calculate_batch_size_and_accumulation_steps(self, model, ddp_model):
        self.logger.info(
            f"calculating batch_size and accumulation_steps "
            f"(effective_batch_size={self.effective_batch_size})"
        )
        # calculate effective_batch_size_per_device
        assert self.effective_batch_size % get_world_size() == 0, \
            f"effective_batch_size ({self.effective_batch_size}) needs to be multiple of " \
            f"world_size ({get_world_size()})"
        effective_batch_size_per_device = calculate_effective_batch_size_per_device(self.effective_batch_size)
        if model.is_batch_size_dependent:
            self.logger.info("model is batch_size dependent -> disabled possible gradient accumulation")
            return effective_batch_size_per_device, 1
        if self.disable_gradient_accumulation:
            self.logger.info(f"gradient accumulation disabled")
            return effective_batch_size_per_device, 1

        self.logger.info(f"effective_batch_size: {self.effective_batch_size}")
        if is_distributed():
            self.logger.info(f"effective_batch_size_per_device: {effective_batch_size_per_device}")
            self.logger.info(f"world_size: {get_world_size()}")

        if self.max_batch_size is None:
            # calculate max_batch_size
            self.logger.info("calculating automatic max_batch_size")
            max_batch_size = calculate_automatic_max_batch_size(
                train_dataset=self.train_dataset,
                collator=self.main_collator,
                # optim step is only taken on (iter_step + 1) % accumulation_steps == 0
                train_step_fn=partial(
                    self.update,
                    model,
                    iter_step=0,
                    accumulation_steps=1,
                    ddp_model=ddp_model,
                ),
                effective_batch_size_per_device=effective_batch_size_per_device,
                device=model.device,
                model=model,
            )
            self.logger.info(f"automatic max_batch_size: {max_batch_size}")
            if is_distributed():
                # check if all devices have the same max_batch_size
                max_batch_sizes = all_gather_nograd(max_batch_size)
                assert all(max_batch_size == mbs for mbs in max_batch_sizes)
        else:
            max_batch_size = calculate_effective_batch_size_per_device(self.max_batch_size)
            self.logger.info(f"using provided max_batch_size {self.max_batch_size} ({max_batch_size} per device)")

        # calculate batch_size and accumulation_steps
        batch_size, accumulation_steps = calculate_batch_size_and_accumulation_steps(
            effective_batch_size_per_device=effective_batch_size_per_device,
            max_batch_size=max_batch_size,
        )
        self.logger.info(f"batch_size: {batch_size}")
        self.logger.info(f"accumulation_steps: {accumulation_steps}")
        return batch_size, accumulation_steps

    def state_dict(self, *args, **kwargs):
        state_dict = dict(state_dict=super().state_dict(*args, **kwargs))

        if is_distributed():
            random_states_per_device = [None for _ in range(get_world_size())]
            all_gather_object(random_states_per_device, get_random_states())
        else:
            random_states_per_device = [get_random_states()]
        callback_state_dicts = [callback.state_dict() for callback in self.callbacks]
        state_dict.update(
            random_states=random_states_per_device,
            epoch=self.update_counter.cur_checkpoint.epoch,
            update=self.update_counter.cur_checkpoint.update,
            sample=self.update_counter.cur_checkpoint.sample,
            callback_state_dicts=callback_state_dicts,
        )
        if isinstance(self.grad_scaler, GradScaler):
            state_dict["grad_scaler"] = self.grad_scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, load_random_states=True):
        # shallow copy
        state_dict = {k: v for k, v in state_dict.items()}
        # load random states
        random_states = state_dict.pop("random_states")
        if load_random_states:
            if len(random_states) != get_world_size():
                # if world_size is different than in the checkpoint the whole resuming run will not be deterministic
                # so don't bother to load any random states
                self.logger.warning(
                    f"trainer checkpoint has different world_size (ckpt_world_size={len(random_states)} "
                    f"world_size={get_world_size()}) -> can't load random states"
                )
            else:
                cur_rank_random_state = random_states[get_rank()]
                set_random_states(**cur_rank_random_state)
        else:
            self.logger.info(f"random states are NOT loaded")

        # load callback state_dicts
        callback_state_dicts = state_dict.pop("callback_state_dicts")
        for callback, sd in zip(self.callbacks, callback_state_dicts):
            callback.load_state_dict(sd)

        # load grad_scaler
        grad_scaler_state_dict = state_dict.pop("grad_scaler", None)
        if isinstance(self.grad_scaler, GradScaler):
            if grad_scaler_state_dict is None:
                self.logger.warning(
                    f"trainer checkpoint has no grad_scaler but current trainer uses {self.precision} precision"
                )
            else:
                self.grad_scaler.load_state_dict(grad_scaler_state_dict)

        # load registered nn.Modules of trainer
        return super().load_state_dict(state_dict=state_dict["state_dict"])

    @property
    def lr_scale_factor(self):
        return self.effective_batch_size

    def _prepare_model(self, model):
        model = model.to(self.device)
        model.initialize(lr_scale_factor=self.lr_scale_factor)
        self.apply_resume_initializer(model)
        return model

    def apply_resume_initializer(self, model):
        # initialize model to state where it was resumed from
        if self.initializer is not None:
            self.logger.info("------------------")
            self.logger.info("loading trainer/model state for resuming")
            assert isinstance(self.initializer, ResumeInitializer)
            self.logger.info(
                f"loading state from checkpoint {self.initializer.stage_id}/"
                f"{self.initializer.stage_name}/{self.initializer.checkpoint}"
            )
            self.initializer.init_trainer(self)
            self.initializer.init_weights(model)
            self.initializer.init_optim(model)

    def get_data_loader(self, periodic_callbacks, batch_size):
        self.logger.info(f"initializing dataloader")
        configs = []
        for c in periodic_callbacks:
            cur_configs, _ = c.register_sampler_configs(self)
            for cur_config in cur_configs:
                if hasattr(cur_config.sampler, "data_source"):
                    dataset_mode = cur_config.sampler.data_source.mode
                else:
                    dataset_mode = "unknown"
                self.logger.info(f"{c} registered {cur_config} dataset_mode='{dataset_mode}'")
            configs += cur_configs
        kwargs = {}
        if self.start_checkpoint.epoch != 0:
            kwargs["start_epoch"] = self.start_checkpoint.epoch
        return self.data_container.get_data_loader(
            main_sampler=self.main_sampler,
            main_collator=self.main_collator,
            batch_size=batch_size,
            epochs=self.end_checkpoint.epoch,
            updates=self.end_checkpoint.update,
            samples=self.end_checkpoint.sample,
            configs=configs,
            **kwargs,
        )

    def wrap_model(self, model):
        assert model.is_initialized, "Model needs to be initialized before DDP wrapping as DPP broadcasts params"
        model = self._wrap_model(model=model)
        trainer_model = self.get_trainer_model(model)
        ddp_model = self.wrap_ddp(trainer_model)
        ddp_model = self.wrap_compile(ddp_model)
        return model, trainer_model, ddp_model

    def get_trainer_model(self, model):
        raise NotImplementedError

    def _wrap_model(self, model):
        return model

    def wrap_ddp(self, model):
        if is_distributed():
            if get_trainable_param_count(model) > 0:
                if self.find_unused_params:
                    self.logger.info(f"using find_unused_params=True")
                    if self.static_graph:
                        self.logger.info(f"using static_graph=True")
                else:
                    assert not self.static_graph
                model = DistributedDataParallel(
                    model,
                    find_unused_parameters=self.find_unused_params,
                    static_graph=self.static_graph,
                )
                if model.device != torch.device("cpu") and self.sync_batchnorm:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            else:
                # DDP broadcasts weights from rank0 to other ranks but raises an error if no param requires grad
                # workaround: temporarily unfreeze one parameter if all parameters are frozen to broadcast weights
                self.logger.info(f"not wrapping into DDP (no trainable parameters) -> only broadcast parameters")
                first_param = next(model.parameters())
                first_param.requires_grad = True
                DistributedDataParallel(model)
                first_param.requires_grad = False
        return model

    def wrap_compile(self, ddp_model):
        if not self.use_torch_compile:
            self.logger.info(f"torch.compile not used (use_torch_compile == False)")
            return ddp_model
        if is_distributed():
            if self.static_graph:
                self.logger.info(f"torch.compile static_graph=True is not supported -> disable torch.compile")
                return ddp_model
        self.logger.info(f"wrapping model with torch.compile")
        return torch.compile(ddp_model)

    def before_training(self):
        pass

    @kp.profile
    def train_model(self, model, callbacks=None):
        model = self._prepare_model(model)
        callbacks = callbacks or self.get_all_callbacks(model=model)
        periodic_callbacks = [callback for callback in callbacks if isinstance(callback, PeriodicCallback)]

        self.before_training()
        model, trainer_model, ddp_model = self.wrap_model(model)
        batch_size, accumulation_steps, train_batches_per_epoch = self._prepare_batch_size(model, ddp_model)
        assert trainer_model.model == model
        # TODO model is moved to GPU seperately from trainer_model because of initializers
        #  -> trainer_model should be moved all at once
        trainer_model = trainer_model.to(model.device)

        data_loader = self.get_data_loader(periodic_callbacks=periodic_callbacks, batch_size=batch_size)
        self.call_before_training(trainer_model=trainer_model, batch_size=batch_size, callbacks=callbacks)
        self._train(
            model=model,
            trainer_model=trainer_model,
            ddp_model=ddp_model,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            data_loader=data_loader,
            train_batches_per_epoch=train_batches_per_epoch,
            periodic_callbacks=periodic_callbacks,
        )
        self.call_after_training(trainer_model=trainer_model, callbacks=callbacks)

    def _train(
            self,
            model,
            trainer_model,
            ddp_model,
            batch_size,
            accumulation_steps,
            data_loader,
            train_batches_per_epoch,
            periodic_callbacks
    ):
        self.logger.info("------------------")
        self.logger.info(f"START TRAINING")

        self.logger.info("initializing dataloader workers")
        with kp.named_profile("iterator"):
            data_iter = iter(data_loader)
        self.logger.info("initialized dataloader workers")

        if self.update_counter.is_finished:
            if not model.is_frozen:
                self.logger.warning("model has optimizer which is not used for evaluation")
            # eval run
            for callback in periodic_callbacks:
                callback.after_epoch(
                    update_counter=self.update_counter,
                    effective_batch_size=self.effective_batch_size,
                    batch_size=batch_size,
                    trainer=self,
                    model=model,
                    trainer_model=trainer_model,
                    data_iter=data_iter,
                )
            for callback in periodic_callbacks:
                callback.after_update(
                    update_counter=self.update_counter,
                    effective_batch_size=self.effective_batch_size,
                    batch_size=batch_size,
                    trainer=self,
                    model=model,
                    trainer_model=trainer_model,
                    data_iter=data_iter,
                )
            CallbackBase.flush()
        else:
            # train run
            is_first_update = True
            while True:
                iter_step = -1
                data_time = 0.
                update_time = 0.
                while True:
                    # check end of epoch
                    remaining_batches = train_batches_per_epoch - (iter_step + 1)
                    if remaining_batches < accumulation_steps:
                        # InterleavedSampler already have the next batches preloaded -> skip them
                        for _ in range(remaining_batches):
                            _ = next(data_iter)
                        break
                    is_last_update_in_epoch = remaining_batches - accumulation_steps < accumulation_steps
                    for callback in periodic_callbacks:
                        callback.before_every_update(update_counter=self.update_counter, model=model)
                    for _ in range(accumulation_steps):
                        # load next batch
                        with kp.named_profile("data_loading"):
                            batch = next(data_iter)
                            iter_step += 1
                        if iter_step % accumulation_steps == 0:
                            model.optim_schedule_step()
                            data_time = 0.
                            update_time = 0.
                        data_time += kp.profiler.last_node.last_time
                        for callback in periodic_callbacks:
                            callback.before_every_accumulation_step(model=model)

                        trainer_model.train()
                        # update contains implicit cuda synchronization points (.detach().cpu(), .item())
                        with kp.named_profile("update"):
                            losses, update_outputs = self.update(
                                batch=batch,
                                iter_step=iter_step,
                                model=model,
                                ddp_model=ddp_model,
                                accumulation_steps=accumulation_steps,
                                periodic_callbacks=periodic_callbacks,
                                is_first_update=is_first_update,
                            )
                        update_time += kp.profiler.last_node.last_time
                        for callback in periodic_callbacks:
                            callback.track_after_accumulation_step(
                                update_counter=self.update_counter,
                                trainer=self,
                                model=model,
                                losses=losses,
                                update_outputs=update_outputs,
                                accumulation_steps=accumulation_steps,
                            )
                        # free references to tensors
                        # noinspection PyUnusedLocal
                        update_outputs = None
                        is_first_update = False

                    # advance counter
                    self.update_counter.add_samples(self.effective_batch_size)
                    self.update_counter.next_update()
                    if is_last_update_in_epoch:
                        self.update_counter.next_epoch()

                    trainer_model.eval()
                    times = dict(data_time=data_time, update_time=update_time)
                    for callback in periodic_callbacks:
                        callback.track_after_update_step(
                            update_counter=self.update_counter,
                            trainer=self,
                            model=model,
                            times=times,
                        )
                    for callback in periodic_callbacks:
                        callback.after_update(
                            update_counter=self.update_counter,
                            effective_batch_size=self.effective_batch_size,
                            batch_size=batch_size,
                            trainer=self,
                            model=model,
                            trainer_model=trainer_model,
                            data_iter=data_iter,
                        )
                    # check end of training
                    if self.update_counter.is_finished:
                        # skip preloaded batches after training when accumulation steps > 1
                        if data_loader.batch_sampler.sampler.epochs is not None:
                            for _ in range(remaining_batches - accumulation_steps):
                                _ = next(data_iter)
                        if data_loader.batch_sampler.sampler.samples is not None:
                            total_batches = int(data_loader.batch_sampler.sampler.samples / batch_size)
                            for _ in range(total_batches % accumulation_steps):
                                _ = next(data_iter)
                        break

                    # no end of epoch -> flush logs from call_after_update
                    if not is_last_update_in_epoch:
                        CallbackBase.flush()

                    # check update/sample based early stopping
                    if self.early_stopper is not None:
                        should_stop_after_update = self.early_stopper.should_stop_after_update(
                            self.update_counter.cur_checkpoint,
                        )
                        if should_stop_after_update:
                            return
                        should_stop_after_sample = self.early_stopper.should_stop_after_sample(
                            self.update_counter.cur_checkpoint,
                            effective_batch_size=self.effective_batch_size,
                        )
                        if should_stop_after_sample:
                            return
                    # update based premature stopping
                    if self.stop_at_update is not None:
                        if self.stop_at_update <= self.update_counter.update:
                            self.logger.info(f"reached stop_at_update (={self.stop_at_update}) -> stop training")
                            return
                    # sample based premature stopping
                    if self.stop_at_sample is not None:
                        if self.stop_at_sample <= self.update_counter.sample:
                            self.logger.info(f"reached stop_at_sample (={self.stop_at_sample}) -> stop training")
                            return

                if self.update_counter.is_full_epoch:
                    for callback in periodic_callbacks:
                        callback.after_epoch(
                            update_counter=self.update_counter,
                            effective_batch_size=self.effective_batch_size,
                            batch_size=batch_size,
                            trainer=self,
                            model=model,
                            trainer_model=trainer_model,
                            data_iter=data_iter,
                        )

                    # check epoch based early stopping
                    if self.early_stopper is not None:
                        if self.early_stopper.should_stop_after_epoch(self.update_counter.cur_checkpoint):
                            return
                    # epoch based premature stopping
                    if self.stop_at_epoch is not None:
                        if self.stop_at_epoch <= self.update_counter.epoch:
                            self.logger.info(f"reached stop_at_epoch (={self.stop_at_epoch}) -> stop training")
                            return
                    CallbackBase.flush()
                # check end of training
                if self.update_counter.is_finished:
                    break
        # check that data_iter was fully consumed
        unconsumed_data_iter_steps = 0
        try:
            next(data_iter)
            self.logger.error("data_iter was not fully consumed -> checking how many batches remain")
            unconsumed_data_iter_steps = 1
            for _ in range(10):
                next(data_iter)
                unconsumed_data_iter_steps += 1
            raise RuntimeError(f"data_iter was not fully consumed (at least {unconsumed_data_iter_steps} unconsumed)")
        except StopIteration:
            if unconsumed_data_iter_steps > 0:
                raise RuntimeError(f"data_iter was not fully consumed ({unconsumed_data_iter_steps} unconsumed)")

    def update(
            self,
            model,
            ddp_model,
            batch,
            iter_step,
            accumulation_steps,
            periodic_callbacks,
            is_first_update=False,
    ):
        model.before_accumulation_step()

        with kp.named_profile_async("forward"):
            with self.autocast_context:
                losses, outputs = ddp_model(batch)
            total_loss = losses["total"] / accumulation_steps

        if not model.is_frozen:
            if self.exit_on_nan_loss and torch.isnan(total_loss):
                self.logger.error(f"encountered NaN loss -> terminate")
                raise RuntimeError("encountered NaN loss")

            # backward
            with kp.named_profile_async("backward"):
                self.grad_scaler.scale(total_loss).backward()
            for callback in periodic_callbacks:
                callback.after_every_backward(total_loss=total_loss)

            # log unused parameters
            if is_first_update:
                unused_param_names = get_paramnames_with_no_gradient(model)
                self.logger.info(f"{len(unused_param_names)} unused parameters")
                for name in unused_param_names:
                    self.logger.info(f"- {name}")

            if (iter_step + 1) % accumulation_steps == 0:
                for callback in periodic_callbacks:
                    callback.before_every_optim_step(model=model, total_loss=total_loss)
                model.optim_step(self.grad_scaler)
                model.optim_zero_grad()

        return {k: v.detach().cpu() for k, v in losses.items()}, outputs

    @property
    def dataset_mode(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        return None

    @staticmethod
    def _init_batchsize_dependent_modules(model, batch_size):
        pass

    def _prepare_batch_size(self, model, ddp_model):
        self.logger.info("------------------")
        self.logger.info("PREPARE TRAINER")
        batch_size, accumulation_steps = self._calculate_batch_size_and_accumulation_steps(model, ddp_model)
        if accumulation_steps > 1 and self.end_checkpoint.update is not None:
            raise NotImplementedError(
                "InterleavedSampler counts every batch as update "
                "-> accumulation steps not supported with update-based end_checkpoint"
            )

        self.config_provider["trainer/batch_size"] = batch_size
        self.config_provider["trainer/accumulation_steps"] = accumulation_steps
        train_batches_per_epoch = int(
            self.main_sampler.effective_length
            / self.effective_batch_size
            * accumulation_steps
        )
        self.logger.info(
            f"train_batches per epoch: {train_batches_per_epoch} "
            f"(world_size={get_world_size()} batch_size={batch_size})"
        )
        self._init_batchsize_dependent_modules(model, batch_size)

        return batch_size, accumulation_steps, train_batches_per_epoch

    def call_before_training(self, trainer_model, batch_size, callbacks):
        self.logger.info("------------------")
        self.logger.info("BEFORE TRAINING")
        trainer_model.eval()
        for c in callbacks:
            c.before_training(
                model=trainer_model.model,
                trainer=self,
                update_counter=self.update_counter,
                trainer_batch_size=batch_size,
            )
        self.logger.info("------------------")
        for callback in callbacks:
            self.logger.info(f"{callback}")

    def call_after_training(self, trainer_model, callbacks):
        self.logger.info("------------------")
        self.logger.info("AFTER TRAINING")
        trainer_model.eval()
        for callback in callbacks:
            callback.after_training(model=trainer_model.model, trainer=self, update_counter=self.update_counter)
        CallbackBase.flush()
