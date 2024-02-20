import einops
import torch

from kappadata.collators import KDSingleCollator
from kappadata.wrappers import ModeWrapper
from torch.utils.data import default_collate
from torch.nn.utils.rnn import pad_sequence



class CfdInterpolatedCollator(KDSingleCollator):
    def collate(self, batch, dataset_mode, ctx=None):
        # make sure that batch was not collated
        assert isinstance(batch, (tuple, list)) and isinstance(batch[0], tuple)
        batch, ctx = zip(*batch)
        # properties in context can have variable shapes (e.g. perm) -> delete ctx
        ctx = {}
        # collect collated properties
        collated_batch = {}

        # query_pos to sparse tensor: batch_size * (num_mesh_points, ndim) -> (batch_size * num_mesh_points, ndim)
        # target to sparse tensor: batch_size * (num_mesh_points, dim) -> (batch_size * num_mesh_points, dim)
        query_pos = []
        query_lens = []
        target = []
        for i in range(len(batch)):
            query_pos_item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="query_pos")
            target_item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="target")
            assert len(query_pos_item) == len(target_item)
            query_lens.append(len(query_pos_item))
            query_pos.append(query_pos_item)
            target.append(target_item)
        assert all(query_lens[0] == query_len for query_len in query_lens[1:])
        collated_batch["query_pos"] = pad_sequence(query_pos, batch_first=True)
        collated_batch["target"] = torch.concat(target)

        # normal collation for other properties (timestep, velocity, geometry2d)
        result = []
        for item in dataset_mode.split(" "):
            if item in collated_batch:
                result.append(collated_batch[item])
            else:
                result.append(
                    default_collate([
                        ModeWrapper.get_item(mode=dataset_mode, batch=sample, item=item)
                        for sample in batch
                    ])
                )

        return tuple(result), ctx

    @property
    def default_collate_mode(self):
        raise RuntimeError

    def __call__(self, batch):
        raise NotImplementedError("wrap KDSingleCollator with KDSingleCollatorWrapper")
