import logging

from providers.summary_providers.base.summary_provider_base import SummaryProviderBase
from providers.summary_providers.noop_summary_provider import NoopSummaryProvider


class SummarySummarizerBase:
    def __init__(self, summary_provider: SummaryProviderBase):
        self.logger = logging.getLogger(type(self).__name__)
        self.summary_provider = summary_provider or NoopSummaryProvider()
