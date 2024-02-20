from .base.summary_provider_base import SummaryProviderBase


class NoopSummaryProvider(SummaryProviderBase):
    def __init__(self):
        super().__init__()
        self.summary = {}

    def update(self, *args, **kwargs):
        self.summary.update(*args, **kwargs)

    def __setitem__(self, key, value):
        self.summary[key] = value

    def __getitem__(self, key):
        return self.summary[key]

    def __contains__(self, key):
        return key in self.summary

    def keys(self):
        return self.summary.keys()

    def get_summary_of_previous_stage(self, stage_name, stage_id):
        return {}

    def flush(self):
        pass

    def summarize_logvalues(self):
        pass
