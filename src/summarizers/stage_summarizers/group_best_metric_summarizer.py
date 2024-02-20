from .base.stage_summarizer_base import StageSummarizerBase
from .best_metric_summarizer import BestMetricSummarizer


class GroupBestMetricSummarizer(StageSummarizerBase):
    def __init__(self, source_group, target_group=None, **kwargs):
        super().__init__(**kwargs)
        self.source_group = source_group
        self.target_group = target_group
        self.kwargs = kwargs

    def summarize(self):
        source_tags = self._get_tags_of_group(self.source_group)

        if self.target_group is not None:
            target_tags = self._get_tags_of_group(self.target_group)
            if len(target_tags) == 0:
                self.logger.warning(f"couldn't find any tags of target_group {self.target_group}")
        else:
            target_tags = []
        for i in range(len(source_tags)):
            kwargs = dict(source_key=source_tags[i])
            if i < len(target_tags):
                kwargs["target_key"] = target_tags[i]
            bms = BestMetricSummarizer(
                **kwargs,
                **self.kwargs,
            )
            bms.summarize()
