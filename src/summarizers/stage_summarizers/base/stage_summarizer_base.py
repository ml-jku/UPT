import logging

import yaml

from providers.path_provider import PathProvider
from providers.summary_providers.base.summary_provider_base import SummaryProviderBase
from providers.summary_providers.noop_summary_provider import NoopSummaryProvider
from utils.checkpoint import Checkpoint


class StageSummarizerBase:
    __all_log_entries = None

    def __init__(self, path_provider: PathProvider, summary_provider: SummaryProviderBase):
        self.logger = logging.getLogger(type(self).__name__)
        self.path_provider = path_provider
        self.summary_provider = summary_provider or NoopSummaryProvider()

    @property
    def all_log_entries(self):
        if StageSummarizerBase.__all_log_entries is None:
            with open(self.path_provider.primitive_output_path / "entries.yaml") as f:
                StageSummarizerBase.__all_log_entries = yaml.safe_load(f)
        return StageSummarizerBase.__all_log_entries

    def summarize(self):
        raise NotImplementedError

    def _checkpoint_from_update(self, update):
        epoch = self.all_log_entries["epoch"][update]
        sample = self.all_log_entries["sample"][update]
        if isinstance(epoch, float) and epoch.is_integer():
            epoch = int(epoch)
        return Checkpoint(epoch=epoch, update=update, sample=sample)

    def _get_tags_of_group(self, group):
        return [key for key in self.all_log_entries.keys() if key.startswith(group)]
