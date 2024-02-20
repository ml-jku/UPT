import kappaconfig as kc

from .testrun_constants import TEST_RUN_EPOCHS, TEST_RUN_UPDATES, TEST_RUN_SAMPLES, TEST_RUN_EFFECTIVE_BATCH_SIZE


class MinDurationPostProcessor(kc.Processor):
    """ limit training duration to a minimum by maniuplating the configuration yaml """

    def preorder_process(self, node, trace):
        if len(trace) == 0:
            return
        parent, parent_accessor = trace[-1]
        if isinstance(parent_accessor, str):
            # trainer
            if parent_accessor == "log_every_n_epochs":
                parent[parent_accessor] = 1
            elif parent_accessor == "log_every_n_updates":
                parent[parent_accessor] = 1
            elif parent_accessor == "log_every_n_samples":
                parent[parent_accessor] = min(parent[parent_accessor], TEST_RUN_EFFECTIVE_BATCH_SIZE)
            elif parent_accessor == "max_epochs":
                parent[parent_accessor] = min(parent[parent_accessor], TEST_RUN_EPOCHS)
            elif parent_accessor == "max_updates":
                parent[parent_accessor] = min(parent[parent_accessor], TEST_RUN_UPDATES)
            elif parent_accessor == "max_samples":
                parent[parent_accessor] = min(parent[parent_accessor], TEST_RUN_SAMPLES)
            # set loggers
            elif parent_accessor == "every_n_epochs":
                parent[parent_accessor] = 1
            elif parent_accessor == "every_n_updates":
                parent[parent_accessor] = 1
            elif parent_accessor == "every_n_samples":
                parent[parent_accessor] = TEST_RUN_EFFECTIVE_BATCH_SIZE
            # initializers
            if parent_accessor == "initializer":
                if parent[parent_accessor]["kind"] == "previous_stage_initializer":
                    self._process_checkpoint(parent[parent_accessor], "checkpoint")
            # schedules
            if "schedule" in parent_accessor:
                for schedule in parent[parent_accessor]:
                    if "start_checkpoint" in schedule:
                        self._process_checkpoint(schedule, "start_checkpoint")
                    if "end_checkpoint" in schedule:
                        self._process_checkpoint(schedule, "end_checkpoint")

    @staticmethod
    def _process_checkpoint(parent, parent_accessor):
        # check if checkpoint is string checkpoint
        if not isinstance(parent[parent_accessor], dict):
            return
        # replace epoch/update/sample checkpoint
        if "epoch" in parent[parent_accessor]:
            parent[parent_accessor] = dict(epoch=TEST_RUN_EPOCHS)
        if "update" in parent[parent_accessor]:
            parent[parent_accessor] = dict(update=TEST_RUN_UPDATES)
        if "sample" in parent[parent_accessor]:
            parent[parent_accessor] = dict(sample=TEST_RUN_SAMPLES)
