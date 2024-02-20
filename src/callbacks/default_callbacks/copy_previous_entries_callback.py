import yaml

from callbacks.base.callback_base import CallbackBase
from initializers.previous_run_initializer import PreviousRunInitializer
from models.base.composite_model_base import CompositeModelBase
from utils.checkpoint import Checkpoint
from utils.model_utils import get_named_models


class CopyPreviousEntriesCallback(CallbackBase):
    @staticmethod
    def _should_include_key(key):
        if key.startswith("profiling/"):
            return False
        return True

    def _before_training(self, model, trainer, **_):
        # collect entries
        all_entries = {}
        for model_name, model in get_named_models(model).items():
            if isinstance(model, CompositeModelBase):
                continue
            for initializer in model.initializers:
                if not isinstance(initializer, PreviousRunInitializer):
                    continue
                entries_uri = self.path_provider.get_primitive_entries_uri(
                    stage_name=initializer.stage_name,
                    stage_id=initializer.stage_id,
                )
                if entries_uri is None:
                    self.logger.info(
                        f"no entries found for initializer of {model_name} (stage_name='{initializer.stage_name}' "
                        f"stage_id={initializer.stage_id}) -> don't copy anything"
                    )
                    continue
                if not entries_uri.exists():
                    self.logger.info(f"entries uri {entries_uri.as_posix()} doesn't exist -> don't copy anything")
                    continue
                with open(entries_uri) as f:
                    entries = yaml.safe_load(f)
                if initializer.stage_name in all_entries:
                    self.logger.info(
                        f"duplicate stage_name when copying entries from {PreviousRunInitializer.__name__} "
                        "-> using first entries"
                    )
                    if entries != all_entries[initializer.stage_name]:
                        self.logger.warning(f"entries are not the same -> only first entries is copied")
                    continue
                all_entries[initializer.stage_name] = entries

        # add to config
        for stage_name, entries in all_entries.items():
            epochs = entries.pop("epoch")
            updates = list(epochs.keys())
            samples = entries.pop("sample")
            for update in updates:
                entry = {
                    f"epoch_{stage_name}": epochs[update],
                    f"update_{stage_name}": update,
                    f"sample_{stage_name}": samples[update],
                }
                entry.update({
                    f"{stage_name}/{key}": value[update]
                    for key, value in entries.items()
                    if self._should_include_key(key)
                })
                self.writer.add_previous_entry(entry)

        if trainer.initializer is not None:
            entries_uri = self.path_provider.get_primitive_entries_uri(
                stage_name=trainer.initializer.stage_name,
                stage_id=trainer.initializer.stage_id,
            )
            if entries_uri is None:
                self.logger.info(
                    f"no entries found for trainer.initializer (stage_name='{trainer.initializer.stage_name}' "
                    f"stage_id={trainer.initializer.stage_id}) -> don't copy anything"
                )
                return
            if not entries_uri.exists():
                self.logger.info(f"entries uri {entries_uri.as_posix()} doesn't exist -> don't copy anything")
                return
            with open(entries_uri) as f:
                entries = yaml.safe_load(f)

            epochs = entries.pop("epoch")
            updates = list(epochs.keys())
            samples = entries.pop("sample")
            if samples[updates[0]] // updates[0] != trainer.effective_batch_size:
                self.logger.warning(
                    f"found different effective_batch_size when resuming trainer "
                    f"(current={trainer.effective_batch_size} old={samples[updates[0]]}) "
                    f"-> don't copy entries"
                )
                return

            for update in updates:
                ckpt = Checkpoint(epoch=epochs[update], update=update, sample=samples[update])
                if ckpt > trainer.start_checkpoint:
                    break
                entry = {
                    f"epoch": epochs[update],
                    f"update": update,
                    f"sample": samples[update],
                }
                entry.update({
                    key: value[update]
                    for key, value in entries.items()
                    if self._should_include_key(key)
                })
                self.writer.add_previous_entry(entry)
