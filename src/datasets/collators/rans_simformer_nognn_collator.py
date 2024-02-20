import torch
from kappadata.collators import KDSingleCollator
from kappadata.wrappers import ModeWrapper
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate


class RansSimformerNognnCollator(KDSingleCollator):
    def collate(self, batch, dataset_mode, ctx=None):
        # make sure that batch was not collated
        assert isinstance(batch, (tuple, list)) and isinstance(batch[0], tuple)
        batch, ctx = zip(*batch)
        # properties in context can have variable shapes (e.g. perm) -> delete ctx
        ctx = {}
        # dict to hold collated items
        collated_batch = {}

        # sparse mesh_pos: batch_size * (num_points, ndim) -> (batch_size * num_points, ndim)
        mesh_pos = []
        mesh_lens = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="mesh_pos")
            mesh_lens.append(len(item))
            mesh_pos.append(item)
        collated_batch["mesh_pos"] = torch.concat(mesh_pos)

        # dense_query_pos: batch_size * (num_points, ndim) -> (batch_size, max_num_points, ndim)
        # sparse target (decoder output is converted to sparse format before loss)
        pressures = [ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="pressure") for sample in batch]
        # predict all positions -> pad
        query_pos = []
        query_lens = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="query_pos")
            assert len(item) == len(pressures[i])
            query_lens.append(len(item))
            query_pos.append(item)
        collated_batch["query_pos"] = pad_sequence(query_pos, batch_first=True)
        collated_batch["pressure"] = torch.concat(pressures).unsqueeze(1)
        # create batch_idx tensor
        batch_size = len(mesh_lens)
        batch_idx = torch.empty(sum(mesh_lens), dtype=torch.long)
        start = 0
        cur_batch_idx = 0
        for i in range(len(mesh_lens)):
            end = start + mesh_lens[i]
            batch_idx[start:end] = cur_batch_idx
            start = end
            cur_batch_idx += 1
        ctx["batch_idx"] = batch_idx
        # create query_batch_idx tensor (required for test loss)
        query_batch_idx = torch.empty(sum(query_lens), dtype=torch.long)
        start = 0
        cur_query_batch_idx = 0
        for i in range(len(query_lens)):
            end = start + query_lens[i]
            query_batch_idx[start:end] = cur_query_batch_idx
            start = end
            cur_query_batch_idx += 1
        ctx["query_batch_idx"] = query_batch_idx
        # create unbatch_idx tensors (unbatch via torch_geometrics.utils.unbatch)
        # e.g. batch_size=2, num_points=[2, 3] -> unbatch_idx=[0, 0, 1, 2, 2, 2] unbatch_select=[0, 2]
        # then unbatching can be done via unbatch(dense, unbatch_idx)[unbatch_select]
        maxlen = max(query_lens)
        unbatch_idx = torch.empty(maxlen * batch_size, dtype=torch.long)
        unbatch_select = []
        unbatch_start = 0
        cur_unbatch_idx = 0
        for i in range(len(query_lens)):
            unbatch_end = unbatch_start + query_lens[i]
            unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
            unbatch_select.append(cur_unbatch_idx)
            cur_unbatch_idx += 1
            unbatch_start = unbatch_end
            padding = maxlen - query_lens[i]
            if padding > 0:
                unbatch_end = unbatch_start + padding
                unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
                cur_unbatch_idx += 1
                unbatch_start = unbatch_end
        unbatch_select = torch.tensor(unbatch_select)
        ctx["unbatch_idx"] = unbatch_idx
        ctx["unbatch_select"] = unbatch_select

        # normal collation for other properties
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
