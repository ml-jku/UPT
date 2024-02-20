import einops
import torch
from kappadata.collators import KDSingleCollator
from kappadata.wrappers import ModeWrapper
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate


class CfdSimformerCollator(KDSingleCollator):
    def __init__(self, num_supernodes=None, **kwargs):
        super().__init__(**kwargs)
        self.num_supernodes = num_supernodes

    def collate(self, batch, dataset_mode, ctx=None):
        # make sure that batch was not collated
        assert isinstance(batch, (tuple, list)) and isinstance(batch[0], tuple)
        batch, ctx = zip(*batch)
        # properties in context can have variable shapes (e.g. perm) -> delete ctx
        ctx = {}
        # collect collated properties
        collated_batch = {}

        # to sparse tensor: batch_size * (num_mesh_points, ndim) -> (batch_size * num_mesh_points, ndim)
        mesh_pos = []
        mesh_lens = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="mesh_pos")
            mesh_lens.append(len(item))
            mesh_pos.append(item)
        collated_batch["mesh_pos"] = torch.concat(mesh_pos)

        # select supernodes
        if self.num_supernodes is not None:
            supernodes_offset = 0
            supernode_idxs = []
            for i in range(len(mesh_lens)):
                perm = torch.randperm(len(mesh_pos[i]))[:self.num_supernodes] + supernodes_offset
                supernode_idxs.append(perm)
                supernodes_offset += mesh_lens[i]
            ctx["supernode_idxs"] = torch.concat(supernode_idxs)

        # create batch_idx tensor
        batch_idx = torch.empty(sum(mesh_lens), dtype=torch.long)
        start = 0
        cur_batch_idx = 0
        for i in range(len(mesh_lens)):
            end = start + mesh_lens[i]
            batch_idx[start:end] = cur_batch_idx
            start = end
            cur_batch_idx += 1
        ctx["batch_idx"] = batch_idx

        # batch_size * (num_mesh_points, num_input_timesteps * num_channels) ->
        # (batch_size * num_mesh_points, num_input_timesteps * num_channels)
        x = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="x")
            assert len(item) == mesh_lens[i]
            x.append(item)
        collated_batch["x"] = torch.concat(x)

        # to sparse tensor: batch_size * (num_grid_points, ndim) -> (batch_size * num_grid_points, ndim)
        grid_pos = []
        grid_lens = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="mesh_pos")
            grid_lens.append(len(item))
            grid_pos.append(item)
        collated_batch["grid_pos"] = torch.concat(grid_pos)

        # create batch_idx tensor
        batch_idx = torch.empty(sum(mesh_lens), dtype=torch.long)
        start = 0
        cur_batch_idx = 0
        for i in range(len(mesh_lens)):
            end = start + mesh_lens[i]
            batch_idx[start:end] = cur_batch_idx
            start = end
            cur_batch_idx += 1
        ctx["batch_idx"] = batch_idx

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
        collated_batch["query_pos"] = pad_sequence(query_pos, batch_first=True)
        collated_batch["target"] = torch.concat(target)

        # create unbatch_idx tensors (unbatch via torch_geometrics.utils.unbatch)
        # e.g. batch_size=2, num_points=[2, 3] -> unbatch_idx=[0, 0, 1, 2, 2, 2] unbatch_select=[0, 2]
        # then unbatching can be done via unbatch(dense, unbatch_idx)[unbatch_select]
        batch_size = len(query_lens)
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

        # sparse mesh_edges:  batch_size * (num_points, ndim) -> (batch_size * num_points, ndim)
        mesh_edges = []
        mesh_edges_offset = 0
        for i in range(len(batch)):
            item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="mesh_edges")
            # if None -> create graph on GPU
            if item is None:
                break
            idx = item + mesh_edges_offset
            mesh_edges.append(idx)
            mesh_edges_offset += mesh_lens[i]
        if len(mesh_edges) > 0:
            # noinspection PyTypedDict
            collated_batch["mesh_edges"] = torch.concat(mesh_edges)
        else:
            collated_batch["mesh_edges"] = None

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
