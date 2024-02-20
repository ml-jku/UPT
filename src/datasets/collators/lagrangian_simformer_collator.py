import einops
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate

from kappadata.collators import KDSingleCollator
from kappadata.wrappers import ModeWrapper
from torch.utils.data import default_collate
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph


class LagrangianSimformerCollator(KDSingleCollator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, batch):
        raise NotImplementedError("wrap KDSingleCollator with KDSingleCollatorWrapper")

    def collate(self, batch, dataset_mode, ctx=None):
        # make sure that batch was not collated
        assert isinstance(batch, (tuple, list)) and isinstance(batch[0], tuple)
        batch, ctx = zip(*batch)
        # extract necessary things from context before deleting it
        num_grid_points = ctx[0].get("num_grid_points", 0)
        for i in range(1, len(ctx)):
            assert ctx[i].get("num_grid_points", 0) == num_grid_points
        time_idx = torch.stack([ctx[i]['time_idx'] for i in range(len(batch))])
        traj_idx = torch.tensor([ctx[i]['traj_idx'] for i in range(len(batch))])
        # properties in context can have variable shapes (e.g. perm) -> delete ctx
        ctx = {}
        # Add idx ctx
        ctx['time_idx'] = time_idx
        ctx['traj_idx'] = traj_idx
        collated_batch = {}
        lens = None
        if ModeWrapper.has_item(mode=dataset_mode, item="x"):
            # create x
            # batch_size * (num_input_timesteps + 1, num_channels, num_points) ->
            # (batch_size * num_points, num_input_timesteps + 1, num_channels)
            x = [ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="x") for sample in batch]
            if lens is None:
                lens = [xx.size(2) for xx in x]
            x_flat = einops.rearrange(torch.concat(x, dim=2), "timesteps channels flat -> flat timesteps channels")
            collated_batch["x"] = x_flat
        else:
            raise NotImplementedError
        pos_items = ("curr_pos", "target_pos_encode")
        for pos_item in pos_items:
            if ModeWrapper.has_item(mode=dataset_mode, item=pos_item):
                # create flat_pos
                # batch_size * (num_points, ndim) -> (batch_size * num_points, ndim) where num_points is variable
                pos = [ModeWrapper.get_item(mode=dataset_mode, batch=sample, item=pos_item) for sample in batch]
                if lens is None:
                    lens = [c.size(0) for c in pos]
                flat_pos = torch.concat(pos)
                collated_batch[pos_item] = flat_pos

        assert ModeWrapper.has_item(mode=dataset_mode, item="edge_index")
        # flatten edge_index
        # batch_size * (num_points, ndim) -> (batch_size * num_points, ndim) where num_points is variable
        edge_index = []
        edge_index_offset = 0
        for i in range(len(batch)):
            idx = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="edge_index") + edge_index_offset
            edge_index.append(idx)
            edge_index_offset += lens[i]
        collated_batch["edge_index"] = torch.concat(edge_index)

        if ModeWrapper.has_item(mode=dataset_mode, item="edge_index_target"):
            edge_index_target = []
            edge_index_offset = 0
            for i in range(len(batch)):
                idx = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="edge_index_target") + edge_index_offset
                edge_index_target.append(idx)
                edge_index_offset += lens[i]
            collated_batch["edge_index_target"] = torch.concat(edge_index_target)

        if ModeWrapper.has_item(mode=dataset_mode, item="edge_features"):
            edge_features = []
            edge_index_offset = 0
            for i in range(len(batch)):
                idx = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="edge_features") + edge_index_offset
                edge_features.append(idx)
                edge_index_offset += lens[i]
            collated_batch["edge_features"] = torch.concat(edge_features)

        if ModeWrapper.has_item(mode=dataset_mode, item="perm"):
            perm_batch = []
            for i in range(len(batch)):
                perm, n_particles = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="perm")
                perm_batch.append(perm + i*n_particles)
            perm_batch = torch.concat(perm_batch)
            collated_batch["perm"] = perm_batch

        if lens is not None and ctx is not None:
            batch_size = len(lens)
            maxlen = max(lens)
            # create batch_idx tensor
            batch_idx = torch.empty(sum(lens), dtype=torch.long)
            start = 0
            cur_batch_idx = 0
            for i in range(len(lens)):
                end = start + lens[i]
                batch_idx[start:end] = cur_batch_idx
                start = end
                cur_batch_idx += 1
            ctx["batch_idx"] = batch_idx
            # unbatch index is used with full data
            target = [ModeWrapper.get_item(mode=dataset_mode, batch=sample, item='target_acc') for sample in batch]
            lens = [xx.size(0) for xx in target]
            maxlen = max(lens)
            unbatch_idx = torch.empty(maxlen * batch_size, dtype=torch.long)
            unbatch_select = []
            unbatch_start = 0
            cur_unbatch_idx = 0
            for i in range(len(lens)):
                unbatch_end = unbatch_start + lens[i]
                unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
                unbatch_select.append(cur_unbatch_idx)
                cur_unbatch_idx += 1
                unbatch_start = unbatch_end
                padding = maxlen - lens[i]
                if padding > 0:
                    unbatch_end = unbatch_start + padding
                    unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
                    cur_unbatch_idx += 1
                    unbatch_start = unbatch_end
            unbatch_select = torch.tensor(unbatch_select)
            ctx["unbatch_idx"] = unbatch_idx
            ctx["unbatch_select"] = unbatch_select

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