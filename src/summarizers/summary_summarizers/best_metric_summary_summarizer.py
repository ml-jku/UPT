import fnmatch

import numpy as np

from utils.infer_higher_is_better import higher_is_better_from_metric_key
from .base.summary_summarizer_base import SummarySummarizerBase


class BestMetricSummarySummarizer(SummarySummarizerBase):
    def __init__(self, pattern, **kwargs):
        super().__init__(**kwargs)
        self.pattern = pattern

    def summarize(self):
        # usually summaries are also generated with corresponding checkpoint info -> filter them out
        filtered_keys = [
            key
            for key in self.summary_provider.keys()
            if "/update" not in key and "/key" not in key
        ]

        matching_keys = []
        # filter out irrelevant keys
        for key in filtered_keys:
            if "*" in self.pattern or "?" in self.pattern:
                # pattern with * or ?
                if not fnmatch.fnmatch(key, self.pattern):
                    continue
            else:
                # pattern with contains
                if self.pattern not in key:
                    continue
            # filter out target metrics "e.g. <target_key>/atbest/<source_key>"
            if "/atbest/" in key:
                continue
            matching_keys.append(key)
        assert len(matching_keys) > 0, f"no matching_keys found for pattern '{self.pattern}'"

        # get best value
        values = [self.summary_provider[key] for key in matching_keys]
        higher_is_better = higher_is_better_from_metric_key(matching_keys[0])
        assert all(higher_is_better == higher_is_better_from_metric_key(key) for key in matching_keys[1:])
        best_value = np.max(values) if higher_is_better else np.min(values)
        best_idxs = np.argwhere(values == best_value).squeeze(1)
        if len(best_idxs) > 1:
            self.logger.info(f"multiple best_idxs {best_idxs}")
        best_idx = best_idxs[0]

        best_key = matching_keys[best_idx]
        self.logger.info(f"pattern={self.pattern} best_key='{best_key}' best_value={best_value}")
        self.summary_provider[f"{self.pattern}/best"] = float(best_value)
        self.summary_provider[f"{self.pattern}/best/key"] = best_key
        # extract source_key from best_key
        # TODO
        # source_key = best_key.replace("best ", "")
        # target_at_source_keys = [key for key in filtered_keys if f" at best {source_key}" in key]
        # for target_at_source_key in target_at_source_keys:
        #     target_at_source_value = self.summary_provider[target_at_source_key]
        #     target_key = target_at_source_key.split(" ")[0]
        #     self.logger.info(f"'{target_at_source_key}' of best {source_key}: {target_at_source_value}")
        #     self.summary_provider[f"{target_key} at best {self.pattern}"] = target_at_source_value
        #     self.summary_provider[f"{target_key} at_best {self.pattern} key"] = target_at_source_key
