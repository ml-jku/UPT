import numpy as np

from utils.infer_higher_is_better import higher_is_better_from_metric_key
from .base.stage_summarizer_base import StageSummarizerBase


class BestMetricSummarizer(StageSummarizerBase):
    """
    looks at the best source_key metric and logs the target_key metric at that global_step
    e.g. source_key="accuracy/valid" target_key="accuracy/test" looks at the best validation accuracy and logs
    the corresponding test accuracy
    target groups can be used to e.g. summarize all kinds of accuracies at once. e.g. with target_key=accuracy it will
    log accuracy/train accuracy/valid accuracy/test
    """

    def __init__(
            self,
            source_key,
            target_key=None,
            target_keys=None,
            target_group=None,
            target_groups=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.source_key = source_key
        self.source_higher_is_better = higher_is_better_from_metric_key(self.source_key)
        # allow setting single or list of target_keys
        self.target_keys = target_keys or []
        if target_key is not None:
            self.target_keys.append(target_key)
        # allow setting single or list of target_groups
        self.target_groups = target_groups or []
        if target_group is not None:
            self.target_groups.append(self.target_groups)

    def summarize(self):
        # log best source metric
        source_updates, source_values = zip(*self.all_log_entries[self.source_key].items())
        best_source_value = np.max(source_values) if self.source_higher_is_better else np.min(source_values)
        best_source_idxs = np.argwhere(source_values == best_source_value).squeeze(1)
        best_source_checkpoints = [self._checkpoint_from_update(source_updates[idx]) for idx in best_source_idxs]
        if len(best_source_checkpoints) > 1:
            self.logger.info(f"multiple best_source_checkpoints {best_source_checkpoints}")
        best_source_idx = best_source_idxs[0]
        self.logger.info(
            f"source_key={self.source_key} target_keys={self.target_keys} target_groups={self.target_groups}"
        )
        self.logger.info(f"best source metric at checkpoint(s) {best_source_checkpoints}: {best_source_value}")

        for target_key in self.target_keys:
            self._log_best_target_metric(target_key, source_updates, best_source_idx)
        for target_group in self.target_groups:
            tags = self._get_tags_of_group(target_group)
            for tag in tags:
                self._log_best_target_metric(tag, source_updates, best_source_idx)

    def _log_best_target_metric(self, target_key, source_updates, best_source_idx):
        target_updates, target_values = zip(*self.all_log_entries[target_key].items())
        # check if source_checkpoints are equal to target_checkpoints (if different log intervals are used for
        # source and target metric the source and target checkpoints won't match)
        assert all(
            self._checkpoint_from_update(source_update) == self._checkpoint_from_update(target_update)
            for source_update, target_update in zip(source_updates, target_updates)
        )
        target_higher_is_better = higher_is_better_from_metric_key(target_key)
        best_target_value = np.max(target_values) if target_higher_is_better else np.min(target_values)
        best_target_idxs = np.argwhere(target_values == best_target_value).squeeze(1)
        best_target_checkpoints = [self._checkpoint_from_update(target_updates[idx]) for idx in best_target_idxs]
        if len(best_target_checkpoints) > 1:
            self.logger.info(f"multiple best_target_checkpoints {best_target_checkpoints}")

        # log target metric at best source metric
        checkpoint = self._checkpoint_from_update(source_updates[best_source_idx])
        target_value = target_values[best_source_idx]
        self.logger.info(f"{target_key} at checkpoint {checkpoint}: {target_value}")
        self.summary_provider[f"{target_key}/atbest/{self.source_key}"] = float(best_target_value)
        self.summary_provider[f"{target_key}/atbest/{self.source_key}/update"] = checkpoint.update
