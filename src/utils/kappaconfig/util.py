import logging
import shutil
from pathlib import Path

import kappaconfig as kc
import yaml

from utils.factory import create_collection
from utils.processors import processor_from_kwargs
from .mindata_postprocessor import MinDataPostProcessor
from .mindata_preprocessor import MinDataPreProcessor
from .minduration_postprocessor import MinDurationPostProcessor
from .minmodel_postprocessor import MinModelPostProcessor
from .minmodel_preprocessor import MinModelPreProcessor
from .none_postprocessor import NonePostProcessor
from .precision_preprocessor import PrecisionPreProcessor
from .remove_large_collections_postprocessor import RemoveLargeCollectionsProcessor
from .schedule_template_postprocessor import ScheduleTemplatePostProcessor


def _get_hp_file_uri(hp_file):
    file_uri = Path(hp_file).expanduser().with_suffix(".yaml")
    assert file_uri.exists(), f"hp_file '{file_uri}' doesn't exist"
    return file_uri


def save_unresolved_hp(hp_file, out_file_uri):
    file_uri = _get_hp_file_uri(hp_file)
    shutil.copy(file_uri, out_file_uri)
    logging.info(f"copied unresolved hp to {out_file_uri}")


def save_resolved_hp(stage_hp, out_file_uri):
    stage_hp = remove_large_collections(stage_hp)
    with open(out_file_uri, "w") as f:
        yaml.safe_dump(stage_hp, f, sort_keys=False)
    logging.info(f"dumped resolved hp to {out_file_uri}")


def get_stage_hp(
        hp_file,
        template_path=None,
        testrun=False,
        minmodelrun=False,
        mindatarun=False,
        mindurationrun=False,
):
    file_uri = _get_hp_file_uri(hp_file)
    run_hp = kc.from_file_uri(file_uri)

    resolver = kc.DefaultResolver(template_path=template_path)
    resolver.pre_processors.append(PrecisionPreProcessor())
    resolver.post_processors.append(NonePostProcessor())
    resolver.post_processors.append(ScheduleTemplatePostProcessor())
    if minmodelrun or testrun:
        resolver.pre_processors.append(MinModelPreProcessor())
        resolver.post_processors.append(MinModelPostProcessor())
    if mindatarun or testrun:
        resolver.pre_processors.append(MinDataPreProcessor())
        resolver.post_processors.append(MinDataPostProcessor())
    if mindurationrun or testrun:
        resolver.post_processors.append(MinDurationPostProcessor())

    resolved = resolver.resolve(run_hp)

    # apply custom processors
    if "processors" in resolved:
        processors = resolved.pop("processors")
        for processor in create_collection(processors, processor_from_kwargs):
            processor(resolved)

    return resolved


def remove_large_collections(stage_hp):
    stage_hp = kc.from_primitive(stage_hp)
    resolver = kc.Resolver(post_processors=[RemoveLargeCollectionsProcessor()])
    resolved = resolver.resolve(stage_hp)
    return resolved


def log_stage_hp(stage_hp):
    stage_hp = remove_large_collections(stage_hp)
    yaml_str = yaml.safe_dump(stage_hp, sort_keys=False)
    # safe_dump appends a trailing newline
    logging.info(f"------------------\n{yaml_str[:-1]}")
