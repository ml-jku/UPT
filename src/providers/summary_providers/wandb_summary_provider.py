import wandb

from .base.summary_provider_base import SummaryProviderBase
from .primitive_summary_provider import PrimitiveSummaryProvider
from ..path_provider import PathProvider


class WandbSummaryProvider(SummaryProviderBase):
    def __init__(self, path_provider: PathProvider):
        super().__init__()
        self.primitive_summary_provider = PrimitiveSummaryProvider(path_provider=path_provider)

    def update(self, *args, **kwargs):
        wandb.run.summary.update(*args, **kwargs)
        self.primitive_summary_provider.update(*args, **kwargs)

    def __setitem__(self, key, value):
        wandb.run.summary[key] = value
        self.primitive_summary_provider[key] = value

    def __getitem__(self, key):
        return self.primitive_summary_provider[key]

    def __contains__(self, key):
        return key in self.primitive_summary_provider

    def keys(self):
        return self.primitive_summary_provider.keys()

    def get_summary_of_previous_stage(self, stage_name, stage_id):
        return self.primitive_summary_provider.get_summary_of_previous_stage(stage_name=stage_name, stage_id=stage_id)

    def flush(self):
        self.primitive_summary_provider.flush()

    def summarize_logvalues(self):
        minmax_dict = self.primitive_summary_provider.summarize_logvalues()
        self.logger.info(f"pushing summarized logvalues to wandb")
        if minmax_dict is not None:
            wandb.run.summary.update(minmax_dict)
