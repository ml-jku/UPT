import logging
import os
from pathlib import Path

import kappaprofiler as kp
import yaml
from torch.distributed import broadcast_object_list
from wandb.util import generate_id

from callbacks.base.callback_base import CallbackBase
from configs.cli_args import CliArgs
from configs.static_config import StaticConfig
from configs.wandb_config import WandbConfig
from datasets import dataset_from_kwargs
from datasets.dummy_dataset import DummyDataset
from distributed.config import is_rank0, is_distributed, get_rank, log_distributed_config
from models import model_from_kwargs
from models.dummy_model import DummyModel
from providers.dataset_config_provider import DatasetConfigProvider
from providers.path_provider import PathProvider
from summarizers.stage_summarizers import stage_summarizer_from_kwargs
from summarizers.summary_summarizers import summary_summarizer_from_kwargs
from trainers import trainer_from_kwargs
from utils.commands import command_from_kwargs
from utils.data_container import DataContainer
from utils.kappaconfig.util import save_unresolved_hp, save_resolved_hp, log_stage_hp
from utils.logging_util import add_global_handlers
from utils.memory_leak_util import get_tensors_in_memory
from utils.seed import set_seed, get_random_int
from utils.system_info import log_system_info, get_cli_command
from utils.version_check import check_versions
from utils.wandb_utils import init_wandb, finish_wandb


def train_stage(stage_hp: dict, static_config: StaticConfig, cli_args: CliArgs, device: str):
    # set environment variables
    for key, value in stage_hp.get("env", {}).items():
        os.environ[key] = value if isinstance(value, str) else str(value)

    # resume
    if cli_args.resume_stage_id is not None:
        assert "initializer" not in stage_hp["trainer"]
        if cli_args.resume_checkpoint is None:
            checkpoint = "latest"
        elif cli_args.resume_checkpoint.startswith("E"):
            checkpoint = dict(epoch=int(cli_args.resume_checkpoint[1:]))
        elif cli_args.resume_checkpoint.startswith("U"):
            checkpoint = dict(update=int(cli_args.resume_checkpoint[1:]))
        elif cli_args.resume_checkpoint.startswith("S"):
            checkpoint = dict(sample=int(cli_args.resume_checkpoint[1:]))
        else:
            # any checkpoint (like cp=last or cp=best.accuracy1.test.main)
            checkpoint = cli_args.resume_checkpoint
        stage_hp["trainer"]["initializer"] = dict(
            kind="resume_initializer",
            stage_id=cli_args.resume_stage_id,
            checkpoint=checkpoint,
        )

    # retrieve stage_id from hp (allows queueing up dependent stages by hardcoding stage_ids in the yamls) e.g.:
    # - pretrain MAE with stageid abcdefgh
    # - finetune MAE where the backbone is initialized with the backbone from stage_id abcdefgh
    stage_id = stage_hp.get("stage_id", None)
    # generate stage_id and sync across devices
    if stage_id is None:
        stage_id = generate_id()
        if is_distributed():
            object_list = [stage_id] if is_rank0() else [None]
            broadcast_object_list(object_list)
            stage_id = object_list[0]
    stage_name = stage_hp.get("stage_name", "default_stage")

    # initialize logging
    path_provider = PathProvider(
        output_path=static_config.output_path,
        model_path=static_config.model_path,
        stage_name=stage_name,
        stage_id=stage_id,
        temp_path=static_config.temp_path,
    )
    message_counter = add_global_handlers(log_file_uri=path_provider.logfile_uri)

    # init seed
    run_name = cli_args.name or stage_hp.pop("name", None)
    seed = stage_hp.pop("seed", None)
    if seed is None:
        seed = 0
        logging.info(f"no seed specified -> using seed={seed}")

    # initialize wandb
    wandb_config_uri = stage_hp.pop("wandb", None)
    if wandb_config_uri == "disabled":
        wandb_mode = "disabled"
    else:
        wandb_mode = cli_args.wandb_mode or static_config.default_wandb_mode
    if wandb_mode == "disabled":
        wandb_config_dict = {}
        if cli_args.wandb_config is not None:
            logging.warning(f"wandb_config is defined via CLI but mode is disabled -> wandb_config is not used")
        if wandb_config_uri is not None:
            logging.warning(f"wandb_config is defined via yaml but mode is disabled -> wandb_config is not used")
    else:
        # retrieve wandb config from yaml
        if wandb_config_uri is not None:
            wandb_config_uri = Path("wandb_configs") / wandb_config_uri
            if cli_args.wandb_config is not None:
                logging.warning(f"wandb_config is defined via CLI and via yaml -> wandb_config from yaml is used")
        # retrieve wandb config from --wandb_config cli arg
        elif cli_args.wandb_config is not None:
            wandb_config_uri = Path("wandb_configs") / cli_args.wandb_config
        # use default wandb_config file
        else:
            wandb_config_uri = Path("wandb_config.yaml")
        with open(wandb_config_uri.with_suffix(".yaml")) as f:
            wandb_config_dict = yaml.safe_load(f)
    wandb_config = WandbConfig(mode=wandb_mode, **wandb_config_dict)
    config_provider, summary_provider = init_wandb(
        device=device,
        run_name=run_name,
        stage_hp=stage_hp,
        wandb_config=wandb_config,
        path_provider=path_provider,
        account_name=static_config.account_name,
        tags=stage_hp.pop("tags", None),
        notes=stage_hp.pop("notes", None),
        group=stage_hp.pop("group", None),
        group_tags=stage_hp.pop("group_tags", None),
    )
    # log codebase "high-level" version name (git commit is logged anyway)
    config_provider["code/mlp"] = "CVSim"
    config_provider["code/tag"] = os.popen("git describe --abbrev=0").read().strip()
    config_provider["code/name"] = "initial"

    # log setup
    logging.info("------------------")
    logging.info(f"stage_id: {stage_id}")
    logging.info(get_cli_command())
    check_versions(verbose=True)
    log_system_info()
    static_config.log()
    cli_args.log()
    log_distributed_config()
    log_stage_hp(stage_hp)
    if is_rank0():
        save_unresolved_hp(cli_args.hp, path_provider.stage_output_path / "hp_unresolved.yaml")
        save_resolved_hp(stage_hp, path_provider.stage_output_path / "hp_resolved.yaml")

    logging.info("------------------")
    logging.info(f"training stage '{path_provider.stage_name}'")
    if is_distributed():
        # using a different seed for every rank to ensure that stochastic processes are different across ranks
        # for large batch_sizes this shouldn't matter too much
        # this is relevant for:
        # - augmentations (augmentation parameters of sample0 of rank0 == augparams of sample0 of rank1 == ...)
        # - the masks of a MAE are the same for every rank
        # NOTE: DDP syncs the parameters in its __init__ method -> same initial parameters independent of seed
        seed += get_rank()
        logging.info(f"using different seeds per process (seed+rank) ")
    set_seed(seed)

    # init datasets
    logging.info("------------------")
    logging.info("initializing datasets")
    datasets = {}
    dataset_config_provider = DatasetConfigProvider(
        global_dataset_paths=static_config.get_global_dataset_paths(),
        local_dataset_path=static_config.get_local_dataset_path(),
        data_source_modes=static_config.get_data_source_modes(),
    )
    if "datasets" not in stage_hp:
        logging.info(f"no datasets found -> initialize dummy dataset")
        datasets["train"] = DummyDataset(
            size=256,
            x_shape=(2,),
            n_classes=2,
        )
    else:
        for dataset_key, dataset_kwargs in stage_hp["datasets"].items():
            logging.info(f"initializing {dataset_key}")
            datasets[dataset_key] = dataset_from_kwargs(
                dataset_config_provider=dataset_config_provider,
                path_provider=path_provider,
                **dataset_kwargs,
            )
    data_container_kwargs = {}
    if "prefetch_factor" in stage_hp:
        data_container_kwargs["prefetch_factor"] = stage_hp.pop("prefetch_factor")
    if "max_num_workers" in stage_hp:
        data_container_kwargs["max_num_workers"] = stage_hp.pop("max_num_workers")
    data_container = DataContainer(
        **datasets,
        num_workers=cli_args.num_workers,
        pin_memory=cli_args.pin_memory,
        config_provider=config_provider,
        seed=get_random_int(),
        **data_container_kwargs,
    )

    # init trainer
    logging.info("------------------")
    logging.info("initializing trainer")
    trainer_kwargs = {}
    if "max_batch_size" in stage_hp:
        trainer_kwargs["max_batch_size"] = stage_hp.pop("max_batch_size")
    trainer = trainer_from_kwargs(
        data_container=data_container,
        device=device,
        sync_batchnorm=cli_args.sync_batchnorm or static_config.default_sync_batchnorm,
        config_provider=config_provider,
        summary_provider=summary_provider,
        path_provider=path_provider,
        **stage_hp["trainer"],
        **trainer_kwargs,
    )
    # register datasets of callbacks (e.g. for ImageNet-C the dataset never changes so its pointless to specify)
    for callback in trainer.callbacks:
        callback.register_root_datasets(
            dataset_config_provider=dataset_config_provider,
            is_mindatarun=cli_args.testrun or cli_args.mindatarun,
        )

    # init model
    logging.info("------------------")
    logging.info("creating model")
    if "model" not in stage_hp:
        logging.info(f"no model defined -> use dummy model")
        model = DummyModel(
            input_shape=trainer.input_shape,
            output_shape=trainer.output_shape,
            update_counter=trainer.update_counter,
            path_provider=path_provider,
            is_frozen=True,
        )
    else:
        model = model_from_kwargs(
            **stage_hp["model"],
            input_shape=trainer.input_shape,
            output_shape=trainer.output_shape,
            update_counter=trainer.update_counter,
            path_provider=path_provider,
            data_container=data_container,
        )
    # logging.info(f"model architecture:\n{model}")
    # moved to trainer as initialization on cuda is different than on cpu
    # model = model.to(stage_config.run_config.device)

    # train model
    trainer.train_model(model)

    # finish callbacks
    CallbackBase.finish()

    # summarize logvalues
    logging.info("------------------")
    logging.info(f"summarize logvalues")
    summary_provider.summarize_logvalues()

    # summarize stage
    if "stage_summarizers" in stage_hp and is_rank0():
        logging.info("------------------")
        logging.info("summarize stage")
        for kwargs in stage_hp["stage_summarizers"]:
            summarizer = stage_summarizer_from_kwargs(
                summary_provider=summary_provider,
                path_provider=path_provider,
                **kwargs,
            )
            summarizer.summarize()
    # summarize summary
    if "summary_summarizers" in stage_hp and is_rank0():
        summary_provider.flush()
        logging.info("------------------")
        for kwargs in stage_hp["summary_summarizers"]:
            summary_summarizer = summary_summarizer_from_kwargs(
                summary_provider=summary_provider,
                **kwargs,
            )
            summary_summarizer.summarize()
    summary_provider.flush()

    # add profiling times to summary_provider
    def try_log_profiler_time(summary_key, profiler_query):
        try:
            summary_provider[summary_key] = kp.profiler.get_node(profiler_query).total_time
        except AssertionError:
            pass

    try_log_profiler_time("profiler/train", "train")
    try_log_profiler_time("profiler/train/iterator", "train.iterator")
    try_log_profiler_time("profiler/train/data_loading", "train.data_loading")
    try_log_profiler_time("profiler/train/update", "train.update")
    try_log_profiler_time("profiler/train/to_device", "train.update.forward.to_device")
    try_log_profiler_time("profiler/train/forward", "train.update.forward")
    try_log_profiler_time("profiler/train/backward", "train.update.backward")
    summary_provider.flush()
    # log profiler times
    logging.info(f"full profiling times:\n{kp.profiler.to_string()}")
    kp.reset()

    # execute commands
    if "on_finish" in stage_hp and is_rank0():
        logging.info("------------------")
        logging.info("ON_FINISH COMMANDS")
        for command in stage_hp["on_finish"]:
            command = command_from_kwargs(**command, stage_id=stage_id)
            # noinspection PyBroadException
            try:
                command.execute()
            except:
                logging.exception(f"failed to execute {command}")

    # cleanup
    logging.info("------------------")
    logging.info(f"CLEANUP")
    data_container.dispose()
    message_counter.log()
    finish_wandb(wandb_config)

    # log how many tensors remain to be aware of potential memory leaks
    all_tensors, cuda_tensors = get_tensors_in_memory()
    logging.info("------------------")
    logging.info(f"{len(all_tensors)} tensors remaining in memory (cpu+gpu)")
    logging.info(f"{len(all_tensors) - len(cuda_tensors)} tensors remaining in memory (cpu)")
    logging.info(f"{len(cuda_tensors)} tensors remaining in memory (gpu)")
