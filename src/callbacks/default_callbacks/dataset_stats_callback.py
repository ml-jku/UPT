import torch
from kappadata.utils.class_counts import get_class_counts
from kappadata.wrappers import ModeWrapper, LabelSmoothingWrapper

from callbacks.base.callback_base import CallbackBase


class DatasetStatsCallback(CallbackBase):
    def _before_training(self, **_):
        for dataset_key, dataset in self.data_container.datasets.items():
            self._log_size(dataset_key, dataset)

    def _log_size(self, dataset_key, dataset):
        self.summary_provider[f"ds_stats/{dataset_key}/len"] = len(dataset)
        self.logger.info(f"{dataset_key}: {len(dataset)} samples")
