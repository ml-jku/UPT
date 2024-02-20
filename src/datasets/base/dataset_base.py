from kappadata.datasets import KDDataset

from providers.dataset_config_provider import DatasetConfigProvider
from providers.path_provider import PathProvider
from utils.collator_from_kwargs import collator_from_kwargs
from utils.factory import create_collection
from utils.param_checking import to_path


class DatasetBase(KDDataset):
    def __init__(
            self,
            collators=None,
            dataset_config_provider: DatasetConfigProvider = None,
            path_provider: PathProvider = None,
            **kwargs,
    ):
        collators = create_collection(collators, collator_from_kwargs)
        super().__init__(collators=collators, **kwargs)
        self.dataset_config_provider = dataset_config_provider
        self.path_provider = path_provider

    def _get_roots(self, global_root, local_root, dataset_identifier):
        # automatically populate global_root/local_root if they are not defined explicitly
        global_root = self._get_global_root(global_root, dataset_identifier)
        if local_root is None:
            if self.dataset_config_provider is not None:
                source_mode = self.dataset_config_provider.get_data_source_mode(dataset_identifier)
                # use local by default
                if source_mode in [None, "local"]:
                    local_root = self.dataset_config_provider.get_local_dataset_path()
        else:
            local_root = to_path(local_root)
        return global_root, local_root

    def _get_global_root(self, global_root, dataset_identifier):
        if global_root is None:
            global_root = self.dataset_config_provider.get_global_dataset_path(dataset_identifier)
        else:
            global_root = to_path(global_root)
        return global_root

    @staticmethod
    def _to_consistent_split(split, has_train=True, has_val=True, has_test=True):
        if has_train and split in ["train", "training"]:
            return "train"
        if has_val and split in ["val", "valid", "validation"]:
            return "val"
        if has_test and split in ["test", "testing"]:
            return "test"
        raise NotImplementedError(
            f"invalid split '{split}' "
            f"(has_train={has_train} has_val={has_val} has_test={has_test})"
        )

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__

    def __len__(self):
        raise NotImplementedError
