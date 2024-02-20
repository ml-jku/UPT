import einops
import scipy
import os
import shutil

import meshio
import numpy as np
import torch
from kappautils.param_checking import to_3tuple, to_2tuple
from torch_geometric.nn.pool import radius, radius_graph

from distributed.config import barrier, is_data_rank0
from .base.dataset_base import DatasetBase


class ShapenetCar(DatasetBase):
    # generated with torch.randperm(889, generator=torch.Generator().manual_seed(0))[:189]
    TEST_INDICES = {
        550, 592, 229, 547, 62, 464, 798, 836, 5, 732, 876, 843, 367, 496,
        142, 87, 88, 101, 303, 352, 517, 8, 462, 123, 348, 714, 384, 190,
        505, 349, 174, 805, 156, 417, 764, 788, 645, 108, 829, 227, 555, 412,
        854, 21, 55, 210, 188, 274, 646, 320, 4, 344, 525, 118, 385, 669,
        113, 387, 222, 786, 515, 407, 14, 821, 239, 773, 474, 725, 620, 401,
        546, 512, 837, 353, 537, 770, 41, 81, 664, 699, 373, 632, 411, 212,
        678, 528, 120, 644, 500, 767, 790, 16, 316, 259, 134, 531, 479, 356,
        641, 98, 294, 96, 318, 808, 663, 447, 445, 758, 656, 177, 734, 623,
        216, 189, 133, 427, 745, 72, 257, 73, 341, 584, 346, 840, 182, 333,
        218, 602, 99, 140, 809, 878, 658, 779, 65, 708, 84, 653, 542, 111,
        129, 676, 163, 203, 250, 209, 11, 508, 671, 628, 112, 317, 114, 15,
        723, 746, 765, 720, 828, 662, 665, 399, 162, 495, 135, 121, 181, 615,
        518, 749, 155, 363, 195, 551, 650, 877, 116, 38, 338, 849, 334, 109,
        580, 523, 631, 713, 607, 651, 168,
    }

    def __init__(
            self,
            split,
            radius_graph_r=None,
            radius_graph_max_num_neighbors=None,
            num_input_points_ratio=None,
            num_query_points_ratio=None,
            grid_resolution=None,
            num_supernodes=None,
            standardize_query_pos=False,
            concat_pos_to_sdf=False,
            global_root=None,
            local_root=None,
            seed=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.split = split
        self.radius_graph_r = radius_graph_r
        self.radius_graph_max_num_neighbors = radius_graph_max_num_neighbors or int(1e10)
        self.num_supernodes = num_supernodes
        self.seed = seed
        if num_input_points_ratio is None:
            self.num_input_points_ratio = None
        else:
            self.num_input_points_ratio = to_2tuple(num_input_points_ratio)
        self.num_query_points_ratio = num_query_points_ratio
        if grid_resolution is not None:
            self.grid_resolution = to_3tuple(grid_resolution)
        else:
            self.grid_resolution = None

        # define spatial min/max of simulation (for normalizing to [0, 1] and then scaling to [0, 200] for pos_embed)
        # min: [-1.7978, -0.7189, -4.2762]
        # max: [1.8168, 4.3014, 5.8759]
        self.domain_min = torch.tensor([-2.0, -1.0, -4.5])
        self.domain_max = torch.tensor([2.0, 4.5, 6.0])
        self.scale = 200
        self.standardize_query_pos = standardize_query_pos
        self.concat_pos_to_sdf = concat_pos_to_sdf

        # mean/std for normalization (calculated on the 700 train samples)
        # import torch
        # from datasets.shapenet_car import ShapenetCar
        # ds = ShapenetCar(global_root="/local00/bioinf/shapenet_car", split="train")
        # targets = [ds.getitem_pressure(i) for i in range(len(ds))]
        # targets = torch.stack(targets)
        # targets.mean()
        # targets.std()
        self.mean = torch.tensor(-36.3099)
        self.std = torch.tensor(48.5743)

        # source_root
        global_root, local_root = self._get_roots(global_root, local_root, "shapenet_car")
        if local_root is None:
            # load data from global_root
            self.source_root = global_root / "preprocessed"
            self.logger.info(f"data_source (global): '{self.source_root}'")
        else:
            # load data from local_root
            self.source_root = local_root / "shapenet_car"
            if is_data_rank0():
                # copy data from global to local
                self.logger.info(f"data_source (global): '{global_root}'")
                self.logger.info(f"data_source (local): '{self.source_root}'")
                if not self.source_root.exists():
                    self.logger.info(
                        f"copying {(global_root / 'preprocessed').as_posix()} "
                        f"to {(self.source_root / 'preprocessed').as_posix()}"
                    )
                    shutil.copytree(global_root / "preprocessed", self.source_root / "preprocessed")
            self.source_root = self.source_root / "preprocessed"
            barrier()
        assert self.source_root.exists(), f"'{self.source_root.as_posix()}' doesn't exist"
        assert self.source_root.name == "preprocessed", f"'{self.source_root.as_posix()}' is not preprocessed folder"

        # discover uris
        self.uris = []
        for i in range(9):
            param_uri = self.source_root / f"param{i}"
            for name in sorted(os.listdir(param_uri)):
                sample_uri = param_uri / name
                if sample_uri.is_dir():
                    self.uris.append(sample_uri)
        assert len(self.uris) == 889, f"found {len(self.uris)} uris instead of 889"
        # split into train/test uris
        if split == "train":
            train_idxs = [i for i in range(len(self.uris)) if i not in self.TEST_INDICES]
            self.uris = [self.uris[train_idx] for train_idx in train_idxs]
            assert len(self.uris) == 700
        elif split == "test":
            self.uris = [self.uris[test_idx] for test_idx in self.TEST_INDICES]
            assert len(self.uris) == 189
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.uris)

    # noinspection PyUnusedLocal
    def getitem_pressure(self, idx, ctx=None):
        p = torch.load(self.uris[idx] / "pressure.th")
        p -= self.mean
        p /= self.std
        return p

    # noinspection PyUnusedLocal
    def getitem_grid_pos(self, idx=None, ctx=None):
        if ctx is not None and "grid_pos" in ctx:
            return ctx["grid_pos"]
        # generate positions for a regular grid (e.g. for GINO encoder)
        assert self.grid_resolution is not None
        x_linspace = torch.linspace(0, self.scale, self.grid_resolution[0])
        y_linspace = torch.linspace(0, self.scale, self.grid_resolution[1])
        z_linspace = torch.linspace(0, self.scale, self.grid_resolution[2])
        # generate positions (grid_resolution[0] * grid_resolution[1], 2)
        meshgrid = torch.meshgrid(x_linspace, y_linspace, z_linspace, indexing="ij")
        grid_pos = torch.stack(meshgrid).flatten(start_dim=1).T
        #
        if ctx is not None:
            assert "grid_pos" not in ctx
            ctx["grid_pos"] = grid_pos
        return grid_pos

    def getitem_mesh_to_grid_edges(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert self.radius_graph_r is not None
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        grid_pos = self.getitem_grid_pos(idx, ctx=ctx)
        # create graph between mesh and regular grid points
        edges = radius(
            x=mesh_pos,
            y=grid_pos,
            r=self.radius_graph_r,
            max_num_neighbors=self.radius_graph_max_num_neighbors,
        ).T
        # edges is (num_points, 2)
        return edges

    def getitem_grid_to_query_edges(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert self.radius_graph_r is not None
        query_pos = self.getitem_query_pos(idx, ctx=ctx)
        grid_pos = self.getitem_grid_pos(idx, ctx=ctx)
        # create graph between mesh and regular grid points
        edges = radius(
            x=grid_pos,
            y=query_pos,
            r=self.radius_graph_r,
            max_num_neighbors=int(1e10),
        ).T
        # edges is (num_points, 2)
        return edges

    def getitem_mesh_pos(self, idx, ctx=None):
        if ctx is not None and "mesh_pos" in ctx:
            return ctx["mesh_pos"]
        mesh_pos = self.getitem_all_pos(idx, ctx=ctx)
        # sample mesh points
        if self.num_input_points_ratio is not None:
            if self.split == "test":
                assert self.seed is not None
            if self.seed is not None:
                # deterministically downsample for evaluation
                generator = torch.Generator().manual_seed(self.seed + int(idx))
            else:
                generator = None
            # get number of samples
            if self.num_input_points_ratio[0] == self.num_input_points_ratio[1]:
                # fixed num_input_points_ratio
                end = int(len(mesh_pos) * self.num_input_points_ratio[0])
            else:
                # variable num_input_points_ratio
                lb, ub = self.num_input_points_ratio
                num_input_points_ratio = torch.rand(size=(1,), generator=generator).item() * (ub - lb) + lb
                end = int(len(mesh_pos) * num_input_points_ratio)
            # uniform sampling
            perm = torch.randperm(len(mesh_pos), generator=generator)[:end]
            mesh_pos = mesh_pos[perm]
        if ctx is not None:
            ctx["mesh_pos"] = mesh_pos
        return mesh_pos

    def getitem_all_pos(self, idx, ctx=None):
        if ctx is not None and "all_pos" in ctx:
            return ctx["all_pos"]
        all_pos = torch.load(self.uris[idx] / "mesh_points.th")
        # rescale for sincos positional embedding
        all_pos.sub_(self.domain_min).div_(self.domain_max - self.domain_min).mul_(self.scale)
        assert torch.all(0 < all_pos)
        assert torch.all(all_pos < self.scale)
        if ctx is not None:
            ctx["all_pos"] = all_pos
        return all_pos

    def getitem_query_pos(self, idx, ctx=None):
        if ctx is not None and "query_pos" in ctx:
            return ctx["query_pos"]
        query_pos = self.getitem_all_pos(idx, ctx=ctx)
        # sample query points
        if self.num_query_points_ratio is not None:
            if self.split == "test":
                assert self.seed is not None
            if self.seed is not None:
                # deterministically downsample for evaluation
                generator = torch.Generator().manual_seed(self.seed + int(idx))
            else:
                generator = None
            # get number of samples
            end = int(len(query_pos) * self.num_query_points_ratio)
            # uniform sampling
            perm = torch.randperm(len(query_pos), generator=generator)[:end]
            query_pos = query_pos[perm]
        # shift query_pos to [-1, 1] (required for torch.nn.functional.grid_sample)
        if self.standardize_query_pos:
            query_pos = query_pos / (self.scale / 2) - 1
        if ctx is not None:
            ctx["query_pos"] = query_pos
        return query_pos

    def _get_generator(self, idx):
        if self.split == "test":
            return torch.Generator().manual_seed(int(idx) + (self.seed or 0))
        if self.seed is not None:
            return torch.Generator().manual_seed(int(idx) + self.seed)
        return None

    # noinspection PyUnusedLocal
    def getitem_mesh_edges(self, idx, ctx=None):
        assert self.radius_graph_r is not None
        # load mesh positions
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        if self.num_supernodes is None:
            # create graph
            edges = radius_graph(
                x=mesh_pos,
                r=self.radius_graph_r,
                max_num_neighbors=self.radius_graph_max_num_neighbors,
                loop=True,
            )
        else:
            # select supernodes
            generator = self._get_generator(idx)
            perm = torch.randperm(len(mesh_pos), generator=generator)[:self.num_supernodes]
            supernodes_pos = mesh_pos[perm]
            # create edges: this can include self-loop or not depending on how many neighbors are found.
            # if too many neighbors are found, neighbors are selected randomly which can discard the self-loop
            edges = radius(
                x=mesh_pos,
                y=supernodes_pos,
                r=self.radius_graph_r,
                max_num_neighbors=self.radius_graph_max_num_neighbors,
            )
            # correct supernode index
            edges[0] = perm[edges[0]]
        return edges.T

    # noinspection PyUnusedLocal
    def getitem_sdf(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert all(self.grid_resolution[0] == grid_resolution for grid_resolution in self.grid_resolution[1:])
        sdf = torch.load(self.uris[idx] / f"sdf_res{self.grid_resolution[0]}.th")
        # check that sdf features were generated with correct positions by checking the distance to the nearest point
        # from the domain minimum/maximum
        # mesh_pos = torch.load(self.uris[idx] / "mesh_points.th")
        # minpoint_dists = (self.domain_min[None, :] - mesh_pos).norm(p=2, dim=1)
        # maxpoint_dists = (self.domain_max[None, :] - mesh_pos).norm(p=2, dim=1)
        # assert torch.allclose(sdf[0, 0, 0], minpoint_dists.min()), f"{sdf[0, 0, 0]} != {minpoint_dists.min()}"
        # assert torch.allclose(sdf[-1, -1, -1], maxpoint_dists.min()), f"{sdf[-1, -1, -1]} != {maxpoint_dists.min()}"
        if self.concat_pos_to_sdf:
            # add position to sdf (GINO uses this for interpolated FNO model)
            x_linspace = torch.linspace(-1, 1, self.grid_resolution[0])
            y_linspace = torch.linspace(-1, 1, self.grid_resolution[1])
            z_linspace = torch.linspace(-1, 1, self.grid_resolution[2])
            grid_pos = torch.meshgrid(x_linspace, y_linspace, z_linspace, indexing="ij")
            # stack features (models expect dim_last format)
            sdf = torch.stack([sdf, *grid_pos], dim=-1)
        else:
            sdf = sdf.unsqueeze(-1)
        return sdf


    def getitem_interpolated(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert self.standardize_query_pos
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        # generate grid positions (these are different than getitem_gridpos because interpolate requires xy indexing)
        # it should be the same if indexing=ij since the mapping and inverse mapping consider the change in indexing
        # but for consistency with scipy.interpolate xy was chosen
        x_linspace = torch.linspace(0, self.scale, self.grid_resolution[0])
        y_linspace = torch.linspace(0, self.scale, self.grid_resolution[1])
        z_linspace = torch.linspace(0, self.scale, self.grid_resolution[2])
        grid_pos = torch.meshgrid(x_linspace, y_linspace, z_linspace, indexing="xy")

        grid = torch.from_numpy(
            scipy.interpolate.griddata(
                mesh_pos.unbind(1),
                torch.ones_like(mesh_pos),
                grid_pos,
                method="linear",
                fill_value=0.,
            ),
        ).float()

        # check for correctness of interpolation
        # import matplotlib.pyplot as plt
        # import os
        # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        # plt.scatter(mesh_pos[:, 0], mesh_pos[:, 1])
        # plt.show()
        # plt.clf()
        # plt.imshow(grid.sum(dim=2).sum(dim=2), origin="lower")
        # plt.show()
        # plt.clf()
        # import torch.nn.functional as F
        # grid = einops.rearrange(grid, "h w d dim -> 1 dim h w d")
        # query_pos = self.getitem_query_pos(idx, ctx=ctx)
        # query_pos = einops.rearrange(query_pos, "num_points ndim -> 1 num_points 1 1 ndim")
        # mesh_values = F.grid_sample(input=grid, grid=query_pos, align_corners=False).squeeze(-1)
        # plt.scatter(*query_pos.squeeze().unbind(1), c=mesh_values[0, 0, :, 0])
        # plt.show()
        # plt.clf()

        return grid