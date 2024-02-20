import os

import einops
import numpy as np
import scipy
import torch
from kappadata.copying.image_folder import copy_imagefolder_from_global_to_local
from kappautils.param_checking import to_2tuple
from torch_geometric.nn.pool import radius, radius_graph

from distributed.config import barrier, is_data_rank0
from utils.num_worker_heuristic import get_fair_cpu_count
from .base.dataset_base import DatasetBase


class CfdDataset(DatasetBase):
    def __init__(
            self,
            version,
            num_input_timesteps,
            radius_graph_r=None,
            radius_graph_max_num_neighbors=None,
            num_input_points=None,
            num_input_points_ratio=None,
            num_input_points_mode="uniform",
            num_supernodes=None,
            supernode_edge_mode="mesh_to_supernode",
            num_query_points=None,
            num_query_points_mode="input",
            couple_query_with_input=False,
            split="train",
            standardize_query_pos=False,
            global_root=None,
            local_root=None,
            grid_resolution=None,
            max_num_sequences=None,
            max_num_timesteps=None,
            norm="mean0std1",
            clamp=None,
            clamp_mode="hard",
            seed=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.version = version
        self.split = split
        self.max_num_sequences = max_num_sequences
        self.max_num_timesteps = max_num_timesteps
        # radius graph
        self.radius_graph_r = radius_graph_r
        self.radius_graph_max_num_neighbors = radius_graph_max_num_neighbors or int(1e10)
        if self.radius_graph_max_num_neighbors == float("inf"):
            self.radius_graph_max_num_neighbors = int(1e10)
        # query
        self.num_query_points = num_query_points
        self.num_query_points_mode = num_query_points_mode
        self.couple_query_with_input = couple_query_with_input
        if couple_query_with_input:
            assert self.num_query_points is None, "couple_query_inputs requires 'num_query_points is None'"
        # num input points
        self.num_input_points = to_2tuple(num_input_points)
        self.num_input_points_ratio = to_2tuple(num_input_points_ratio)
        self.num_input_points_mode = num_input_points_mode
        self.num_supernodes = num_supernodes
        self.supernode_edge_mode = supernode_edge_mode
        assert not (self.num_input_points is not None and self.num_input_points_ratio is not None)
        # grid
        assert grid_resolution is None or len(grid_resolution) == 2
        self.grid_resolution = grid_resolution
        # input timesteps
        assert 0 < num_input_timesteps
        self.num_input_timesteps = num_input_timesteps
        # standardize query pos for interpolated (for torch.nn.functional.grid_sample)
        self.standardize_query_pos = standardize_query_pos
        self.seed = seed
        self.norm = norm
        self.clamp = clamp
        self.clamp_mode = clamp_mode
        self.num_input_points_cache = []
        if norm == "none":
            self.mean = torch.tensor([0., 0., 0.])
            self.std = torch.tensor([1., 1., 1.])
        elif version == "v1-1sim":
            self.mean = torch.tensor([0.029124850407242775, 0.00255209649913013, 0.0010001148330047727])
            self.std = torch.tensor([0.026886435225605965, 0.01963668502867222, 0.001666962169110775])
        elif version == "v1-2sims":
            # NOTE: copied from v1-1sim
            self.mean = torch.tensor([0.029124850407242775, 0.00255209649913013, 0.0010001148330047727])
            self.std = torch.tensor([0.026886435225605965, 0.01963668502867222, 0.001666962169110775])
        elif version == "v1-1000sims":
            self.mean = torch.tensor([0.036486752331256866, 2.509498517611064e-05, 0.000451924919616431])
            self.std = torch.tensor([0.026924047619104385, 0.02058381214737892, 0.002078353427350521])
        elif version == "v1-10000sims":
            # crashed cases: case_6709 case_3580 case_3577 case_3578
            # incomplete cases: case_3581
            # no mesh data: case_3579
            self.mean = torch.tensor([0.0152587890625, -1.7881393432617188e-06, 0.0003612041473388672])
            self.std = torch.tensor([0.0233612060546875, 0.0184173583984375, 0.0019378662109375])
        elif version == "v1-686sims-1object":
            self.mean = torch.tensor([0.03460693359375, -3.236532211303711e-05, 7.647275924682617e-05])
            self.std = torch.tensor([0.01055145263671875, 0.00829315185546875, 0.0004229545593261719])
        elif version == "v1-1900sims":
            self.mean = torch.tensor([0.03460693359375, -1.806020736694336e-05, 0.00010699033737182617])
            self.std = torch.tensor([0.01363372802734375, 0.01102447509765625, 0.0006461143493652344])
        elif version == "v2-2500sims":
            if norm == "mean0std1q25":
                # cfddataset_norm.py --q 0.25 --root /local00/bioinf/mesh_dataset/v2-2500sims --exclude_last 500
                self.mean = torch.tensor([0.03450389206409454, -5.949020305706654e-06, 0.00010136327182408422])
                self.std = torch.tensor([0.0031622101087123156, 0.0018765029963105917, 0.0001263884623767808])
            else:
                raise NotImplementedError
        elif version == "v2-5000sims":
            if norm == "mean0std1":
                self.mean = torch.tensor([0.0258941650390625, -1.823902130126953e-05, 0.00012934207916259766])
                self.std = torch.tensor([0.01482391357421875, 0.01200103759765625, 0.0007719993591308594])
            elif norm == "mean0std1q05":
                # cfddataset_norm.py with --q 0.05
                self.mean = torch.tensor([0.035219039767980576, -2.1968364308122545e-05, 0.0001966722047654912])
                self.std = torch.tensor([0.010309861041605473, 0.007318499963730574, 0.0005381687660701573])
            elif norm == "mean0std1q1":
                # cfddataset_norm.py with --q 0.1
                self.mean = torch.tensor([0.036569397896528244, -2.364995816606097e-05, 0.00019191036699339747])
                self.std = torch.tensor([0.00839781854301691, 0.005956545472145081, 0.0004608448361977935])
            elif norm == "mean0std1q25":
                # cfddataset_norm.py with --q 0.25
                self.mean = torch.tensor([0.036188144236803055, -2.3106376829673536e-05, 0.0001511715818196535])
                self.std = torch.tensor([0.0047589014284312725, 0.0034182844683527946, 0.00027269049314782023])
            else:
                raise NotImplementedError
        elif version == "v2-6000sims":
            if norm == "mean0std1q25":
                # python cfddataset_norm.py --root .../v2-6000sims --q 0.25 --exclude_last 1000
                self.mean = torch.tensor([0.026319274678826332, -1.2412071725975693e-07, 5.59896943741478e-05])
                self.std = torch.tensor([0.0031868498772382736, 0.0021304511465132236, 0.000102771315141581])
            else:
                raise NotImplementedError
        elif version == "v3-10000sims":
            # crashed cases: case_1589 case_2188 case_2679 case_3021 case_5378 case_7508 case_7644 case_8035 case_8757
            if norm == "mean0std1q25":
                #cfddataset_norm.py --root .../v3-10000sims --q 0.25 --exclude_last 2000
                self.mean = torch.tensor([0.03648518770933151, 1.927249059008318e-06, 0.000112384237581864])
                self.std = torch.tensor([0.005249467678368092, 0.003499444341287017, 0.0002817418717313558])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # define spatial min/max of simulation
        # x in [-0.5, 0.5] y in [-0.5, 1] -> sale by 200 -> x in [0, 200] y in [0, 300])
        self.max_x_pos = 200
        self.max_y_pos = 300
        self.pos_scale = 200
        self.sim_x_pos_min = -0.5
        self.sim_y_pos_min = -0.5

        # source_root
        global_root, local_root = self._get_roots(global_root, local_root, "mesh_dataset")
        if local_root is None:
            # load data from global_root
            self.source_root = global_root / version
            self.logger.info(f"data_source (global): '{self.source_root}'")
        else:
            # load data from local_root
            self.source_root = local_root / "mesh_dataset"
            if is_data_rank0():
                # copy data from global to local
                self.logger.info(f"data_source (global): '{global_root}'")
                self.logger.info(f"data_source (local): '{self.source_root}'")
                # copy images
                copy_imagefolder_from_global_to_local(
                    global_path=global_root,
                    local_path=self.source_root,
                    relative_path=version,
                    log_fn=self.logger.info,
                    num_workers=min(10, get_fair_cpu_count()),
                )
            self.source_root = self.source_root / version
            barrier()
        assert self.source_root.exists(), f"'{self.source_root.as_posix()}' doesn't exist"

        # load name of sequences (name of folders)
        seqnames = list(sorted([name for name in os.listdir(self.source_root) if (self.source_root / name).is_dir()]))
        assert len(seqnames) > 0, f"couldnt find any sequences in '{self.source_root.as_posix()}'"
        # filter out seqnames of split
        seqnames = self._filter_split_seqnames(seqnames)
        assert len(seqnames) > 0, f"filtered out all sequences of '{self.source_root.as_posix()}'"
        # load filenames for each sequence
        self.samples = []
        for seqname in seqnames:
            samples = [
                fname
                for fname in sorted(os.listdir(self.source_root / seqname))
                if self._is_timestep_fname(fname)
            ]
            self.samples.append((seqname, samples))
        # check that all folders have equally many timesteps
        seqlens = [len(fnames) for _, fnames in self.samples]
        if not all(seqlens[0] == seqlen for seqlen in seqlens):
            for seqname, fnames in self.samples:
                self.logger.info(f"- {seqname} {len(fnames)}")
            raise RuntimeError("not all sequencelengths are the same")
        if self.max_num_timesteps is not None:
            assert max_num_timesteps <= seqlens[0]
            self.max_timestep = max_num_timesteps
        else:
            self.max_timestep = seqlens[0]
        # first timestep cannot be predicted
        self.max_timestep -= 1

    def _filter_split_seqnames(self, seqnames):
        if self.version in ["v1-1sim"]:
            assert len(seqnames) == 1
            return seqnames
        if self.version in ["v1-2sims"]:
            assert len(seqnames) == 2
            return seqnames
        if self.version in ["v1-1000sims"]:
            assert self.max_num_sequences is None
            # seqnames is e.g. "case_95"
            seqname_to_caseidx = {seqname: int(seqname.split("_")[1]) for seqname in seqnames}
            if self.split == "train":
                return [seqname for seqname, idx in seqname_to_caseidx.items() if 10 <= idx]
            if self.split == "test":
                return [seqname for seqname, idx in seqname_to_caseidx.items() if idx < 10]
            if self.split == "train-10sims":
                return [seqname for seqname, idx in seqname_to_caseidx.items() if 10 <= idx][:10]
        if self.version in ["v1-10000sims"]:
            assert self.max_num_sequences is None
            # seqnames is e.g. "case_95"
            seqname_to_caseidx = {seqname: int(seqname.split("_")[1]) for seqname in seqnames}
            if self.split == "train":
                return [seqname for seqname, idx in seqname_to_caseidx.items() if idx <= 10005]
            if self.split == "test":
                return [seqname for seqname, idx in seqname_to_caseidx.items() if idx > 10005]
            if self.split == "train-10sims":
                return [seqname for seqname, idx in seqname_to_caseidx.items() if idx < 10]
        if self.version in ["v1-686sims-1object"]:
            caseidx_to_seqname = {int(seqname.split("_")[1]): seqname for seqname in seqnames}
            sorted_caseidxs = list(sorted(caseidx_to_seqname.keys()))
            num_train_sequences = 650
            if self.split == "train":
                split_seqnames = [caseidx_to_seqname[case_idx] for case_idx in sorted_caseidxs[:num_train_sequences]]
            elif self.split == "test":
                split_seqnames = [caseidx_to_seqname[case_idx] for case_idx in sorted_caseidxs[num_train_sequences:]]
            else:
                raise NotImplementedError
            if self.max_num_sequences is not None:
                split_seqnames = split_seqnames[:self.max_num_sequences]
            return split_seqnames
        if self.version in ["v1-1900sims"]:
            caseidx_to_seqname = {int(seqname.split("_")[1]): seqname for seqname in seqnames}
            sorted_caseidxs = list(sorted(caseidx_to_seqname.keys()))
            num_train_sequences = 1900
            if self.split == "train":
                split_seqnames = [caseidx_to_seqname[case_idx] for case_idx in sorted_caseidxs[:num_train_sequences]]
            elif self.split == "test":
                split_seqnames = [caseidx_to_seqname[case_idx] for case_idx in sorted_caseidxs[num_train_sequences:]]
            else:
                raise NotImplementedError
            if self.max_num_sequences is not None:
                split_seqnames = split_seqnames[:self.max_num_sequences]
            return split_seqnames
        if self.version == "v2-5000sims":
            # v2-5000sims is a subset of v2-10000sims
            caseidx_to_seqname = {int(seqname.split("_")[1]): seqname for seqname in seqnames}
            sorted_caseidxs = list(sorted(caseidx_to_seqname.keys()))
            assert len(sorted_caseidxs) >= 5500
            num_train_sequences = 5000
            num_test_sequences = 500
            num_val_sequences = 500
            if self.split == "train":
                split_seqnames = [caseidx_to_seqname[case_idx] for case_idx in sorted_caseidxs[:num_train_sequences]]
            elif self.split == "test":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[num_train_sequences:num_train_sequences + num_test_sequences]
                ]
            elif self.split == "val":
                start = num_train_sequences + num_test_sequences
                end = start + num_val_sequences
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[start:end]
                ]
            else:
                raise NotImplementedError
            if self.max_num_sequences is not None:
                split_seqnames = split_seqnames[:self.max_num_sequences]
            return split_seqnames
        if self.version == "v2-10000sims":
            caseidx_to_seqname = {int(seqname.split("_")[1]): seqname for seqname in seqnames}
            sorted_caseidxs = list(sorted(caseidx_to_seqname.keys()))
            assert len(sorted_caseidxs) >= 10000
            num_train_sequences = 9500
            num_test_sequences = 500
            if self.split == "train":
                split_seqnames = [caseidx_to_seqname[case_idx] for case_idx in sorted_caseidxs[:num_train_sequences]]
            elif self.split == "test":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[num_train_sequences:num_train_sequences + num_test_sequences]
                ]
            else:
                raise NotImplementedError
            if self.max_num_sequences is not None:
                split_seqnames = split_seqnames[:self.max_num_sequences]
            return split_seqnames
        if self.version == "v2-2500sims":
            # v2-2500sims is a subset of v2-10000sims of simulations that contain only 1 object
            caseidx_to_seqname = {int(seqname.split("_")[1]): seqname for seqname in seqnames}
            sorted_caseidxs = list(sorted(caseidx_to_seqname.keys()))
            assert len(sorted_caseidxs) >= 2500
            num_train_sequences = 2000
            num_test_sequences = 500
            if self.split == "train":
                split_seqnames = [caseidx_to_seqname[case_idx] for case_idx in sorted_caseidxs[:num_train_sequences]]
            elif self.split == "test":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[num_train_sequences:num_train_sequences + num_test_sequences]
                ]
            else:
                raise NotImplementedError
            if self.max_num_sequences is not None:
                split_seqnames = split_seqnames[:self.max_num_sequences]
            return split_seqnames
        if self.version == "v2-6000sims":
            # v2-6000sims is subset of v2-10000sims of simulations with 0.01 < v < 0.04
            caseidx_to_seqname = {int(seqname.split("_")[1]): seqname for seqname in seqnames}
            sorted_caseidxs = list(sorted(caseidx_to_seqname.keys()))
            assert len(sorted_caseidxs) == 6000
            num_train_sequences = 5000
            num_test_sequences = 1000
            if self.split == "train":
                split_seqnames = [caseidx_to_seqname[case_idx] for case_idx in sorted_caseidxs[:num_train_sequences]]
            elif self.split == "test":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[num_train_sequences:num_train_sequences + num_test_sequences]
                ]
            else:
                raise NotImplementedError
            if self.max_num_sequences is not None:
                split_seqnames = split_seqnames[:self.max_num_sequences]
            return split_seqnames
        if self.version == "v3-10000sims":
            caseidx_to_seqname = {int(seqname.split("_")[1]): seqname for seqname in seqnames}
            sorted_caseidxs = list(sorted(caseidx_to_seqname.keys()))
            assert len(sorted_caseidxs) == 10000
            num_train_sequences = 8000
            num_valid_sequences = 1000
            num_test_sequences = 1000
            if self.split == "train":
                split_seqnames = [caseidx_to_seqname[case_idx] for case_idx in sorted_caseidxs[:num_train_sequences]]
                assert len(split_seqnames) == num_train_sequences
            elif self.split == "valid":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[num_train_sequences:num_train_sequences + num_valid_sequences]
                ]
                assert len(split_seqnames) == num_valid_sequences
            elif self.split == "test":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[num_train_sequences + num_valid_sequences:]
                ]
                assert len(split_seqnames) == num_test_sequences
            else:
                raise NotImplementedError
            if self.max_num_sequences is not None:
                split_seqnames = split_seqnames[:self.max_num_sequences]
            return split_seqnames
        raise NotImplementedError

    @staticmethod
    def _is_timestep_fname(fname):
        if not fname.endswith(".th"):
            return False
        if fname in ["object_mask.th", "U_init.th", "x.th", "y.th", "movement_per_position.th", "num_objects.th"]:
            return False
        if fname.startswith("edge_index"):
            return False
        if fname.startswith("sampling_weights"):
            return False
        assert fname.endswith("_mesh.th") and fname[:-len("_mesh.th")].isdigit()
        return True

    def __len__(self):
        if self.num_input_timesteps == float("inf"):
            # dataset for rollout: all samples of a sequence are returned
            return len(self.samples)
        return self.max_timestep * len(self.samples)

    # noinspection PyUnusedLocal
    def getitem_timestep(self, idx, ctx=None):
        return idx % self.max_timestep

    def getshape_timestep(self):
        return self.max_timestep,

    def denormalize(self, data, inplace=False):
        assert data.size(1) == len(self.mean)
        shape = [1] * data.ndim
        shape[1] = len(self.mean)
        mean = self.mean.view(*shape).to(data.device)
        std = self.std.view(*shape).to(data.device)
        if inplace:
            data.mul_(std).add_(mean)
        else:
            data = data * std + mean
        return data

    def _get_sim_name(self, idx):
        if self.num_input_timesteps == float("inf"):
            # dataset for rollout -> return full trajectory
            sim_name, _ = self.samples[idx]
        else:
            # dataset for training -> predict random timestep
            seqidx = idx // self.max_timestep
            sim_name, _ = self.samples[seqidx]
        return sim_name

    # noinspection PyUnusedLocal
    def getitem_geometry2d(self, idx, ctx=None):
        sim_name = self._get_sim_name(idx)
        return torch.load(self.source_root / sim_name / "object_mask.th")

    def getshape_geometry2d(self):
        # none=0
        # is_obstacle=1
        shape = self.getitem_geometry2d(0).shape
        return 2, *shape

    # noinspection PyUnusedLocal
    def getitem_num_objects(self, idx, ctx=None):
        sim_name = self._get_sim_name(idx)
        return torch.load(self.source_root / sim_name / f"num_objects.th")

    # noinspection PyUnusedLocal
    def getitem_velocity(self, idx, ctx=None):
        if self.version in ["v1-1sim"]:
            return 0
        sim_name = self._get_sim_name(idx)
        if self.version in ["v1-2sims"]:
            if idx < len(self) // 2:
                return 0
            return 1
        # v is samples from U[0.01, 0.06]
        v = torch.load(self.source_root / sim_name / f"U_init.th")
        # convert to [0, 200] for sincos embedding
        v.sub_(0.01).div_(0.05).mul_(200)
        return v

    def _get_generator(self, idx):
        if self.num_input_timesteps == float("inf"):
            # deterministically downsample for evaluation
            return torch.Generator().manual_seed(int(idx) + (self.seed or 0))
        if self.split == "test":
            assert self.seed is not None
        if self.seed is not None:
            return torch.Generator().manual_seed(int(idx) + self.seed)
        return None

    def _downsample_input(self, data, idx=None, ctx=None):
        if self.num_input_points_ratio is None and self.num_input_points is None:
            return data
        assert ctx is not None
        if "input_perm" in ctx:
            perm = ctx["input_perm"]
        else:
            generator = self._get_generator(idx)
            if self.num_input_points is not None:
                if self.num_input_points[0] == self.num_input_points[1]:
                    # fixed num_input_points
                    end = self.num_input_points[0]
                else:
                    # variable num_input_points
                    # make sure each batch has the same number of points to avoid heavy memory fluctuations
                    # this is ensured by ensuring that the sum of points in 2 consecutive samples is lb + ub
                    if len(self.num_input_points_cache) > 0:
                        assert len(self.num_input_points_cache) == 1
                        end = self.num_input_points_cache.pop()
                    else:
                        assert generator is None, "variable num_input_points doesnt support seed"
                        lb, ub = self.num_input_points
                        midpoint = torch.randint(ub - lb, size=(1,)).item()
                        end = ub - midpoint
                        self.num_input_points_cache.append(lb + midpoint)
            elif self.num_input_points_ratio is not None:
                if self.num_input_points_ratio[0] == self.num_input_points_ratio[1]:
                    # fixed num_input_points_ratio
                    end = int(len(data) * self.num_input_points_ratio[0])
                else:
                    # variable num_points_ratio
                    lb, ub = self.num_input_points_ratio
                    num_points_ratio = torch.rand(size=(1,), generator=generator).item() * (ub - lb) + lb
                    end = int(len(data) * num_points_ratio)
            else:
                raise NotImplementedError

            if self.num_input_points_mode == "uniform":
                # uniform sampling
                perm = torch.randperm(len(data), generator=generator)[:end]
            else:
                # weighted sampling
                sim_name = self._get_sim_name(idx)
                weights = torch.load(self.source_root / sim_name / f"sampling_weights_{self.num_input_points_mode}.th")
                perm = torch.multinomial(weights.float(), num_samples=end, replacement=False, generator=generator)
            ctx["input_perm"] = perm
        return data[perm]

    def _downsample_query(self, data, idx=None, ctx=None):
        # rollout needs to have same permutation for input and query because the prediction is used as next input
        # if rollout is via latent space its not strictly needed but this edge case is not considered
        if self.num_input_timesteps == float("inf"):
            assert self.num_query_points is None
            return self._downsample_input(data, idx=idx, ctx=ctx)
        # use input perm also for query (required for consistency losses)
        if self.couple_query_with_input:
            assert self.num_query_points is None
            return self._downsample_input(data, idx=idx, ctx=ctx)
        if self.num_query_points is None:
            return data
        if "query_perm" in ctx:
            perm = ctx["query_perm"]
        else:
            if self.num_query_points_mode == "input":
                # use the same permutation as for downsampling the input
                # -> the points that are used as input are always used as target
                perm = ctx["input_perm"]
                assert len(perm) >= self.num_query_points
            elif self.num_query_points_mode == "arbitrary":
                # generate new permutation -> any points can be used as target
                generator = self._get_generator(idx)
                perm = torch.randperm(len(data), generator=generator)
            else:
                raise NotImplementedError
            perm = perm[:self.num_query_points]
            ctx["query_perm"] = perm
        return data[perm]

    def _downsample_reconstruction_output(self, data, idx=None, ctx=None):
        # rollout needs to have same permutation for input and query because the prediction is used as next input
        # if rollout is via latent space its not strictly needed but this edge case is not considered
        assert not self.num_input_timesteps == float("inf")
        assert not self.couple_query_with_input
        if self.num_query_points is None:
            return data
        if "rec_perm" in ctx:
            perm = ctx["rec_perm"]
        else:
            if self.num_query_points_mode == "input":
                # use the same permutation as for downsampling the input
                # -> the points that are used as input are always used as target
                raise NotImplementedError
            elif self.num_query_points_mode == "arbitrary":
                # generate new permutation -> any points can be used as target
                generator = self._get_generator(idx)
                perm = torch.randperm(len(data), generator=generator)
            else:
                raise NotImplementedError
            perm = perm[:self.num_query_points]
            ctx["rec_perm"] = perm
        return data[perm]

    def _load_xy(self, case_uri):
        # swap from simulation format (width=x height=y) to torch format (height=x width=y)
        x = torch.load(case_uri / f"y.th").float()
        y = torch.load(case_uri / f"x.th").float()
        # shift positions to start from 0 and scale by 200
        # x in [-0.5, 0.5] y in [-0.5, 1]
        x.sub_(self.sim_x_pos_min).mul_(self.pos_scale)
        y.sub_(self.sim_y_pos_min).mul_(self.pos_scale)
        assert torch.all(0 <= x), f"error in {sim_name} x.min={x.min().item()}"
        assert torch.all(x < self.max_x_pos), f"error in {sim_name} y.max={x.max().item()}"
        assert torch.all(0 <= y), f"error in {sim_name} y.min={y.min().item()}"
        assert torch.all(y < self.max_y_pos), f"error in {sim_name} y.max={y.max().item()}"
        # stack
        all_pos = torch.stack([x, y], dim=1)
        return all_pos

    def getitem_all_pos(self, idx, ctx=None):
        if ctx is not None and "all_pos" in ctx:
            return ctx["all_pos"]
        sim_name = self._get_sim_name(idx)
        all_pos = self._load_xy(self.source_root / sim_name)
        # cache
        if ctx is not None:
            assert "all_pos" not in ctx
            ctx["all_pos"] = all_pos
        return all_pos

    def getitem_mesh_pos(self, idx, ctx=None):
        if ctx is not None and "mesh_pos" in ctx:
            return ctx["mesh_pos"]
        mesh_pos = self.getitem_all_pos(idx, ctx=ctx)
        mesh_pos = self._downsample_input(mesh_pos, idx=idx, ctx=ctx)
        if ctx is not None:
            assert "mesh_pos" not in ctx
            ctx["mesh_pos"] = mesh_pos
        return mesh_pos

    def getitem_query_pos(self, idx, ctx=None):
        if ctx is not None and "query_pos" in ctx:
            return ctx["query_pos"]
        query_pos = self.getitem_all_pos(idx, ctx=ctx)
        query_pos = self._downsample_query(query_pos, idx=idx, ctx=ctx)
        if self.standardize_query_pos:
            query_pos = query_pos / (torch.tensor([self.max_x_pos, self.max_y_pos])[None, :] / 2) - 1
        if ctx is not None:
            assert "query_pos" not in ctx
            ctx["query_pos"] = query_pos
        return query_pos

    def getitem_reconstruction_pos(self, idx, ctx=None):
        if ctx is not None and "rec_pos" in ctx:
            return ctx["rec_pos"]
        rec_pos = self.getitem_all_pos(idx, ctx=ctx)
        rec_pos = self._downsample_reconstruction_output(rec_pos, idx=idx, ctx=ctx)
        if ctx is not None:
            assert "rec_pos" not in ctx
            ctx["rec_pos"] = rec_pos
        return rec_pos

    # noinspection PyUnusedLocal
    def getitem_grid_pos(self, idx=None, ctx=None):
        if ctx is not None and "grid_pos" in ctx:
            return ctx["grid_pos"]
        # generate positions for a regular grid (e.g. for GINO encoder)
        assert self.grid_resolution is not None
        x_linspace = torch.linspace(0, self.max_x_pos, self.grid_resolution[0])
        y_linspace = torch.linspace(0, self.max_y_pos, self.grid_resolution[1])
        # generate positions (grid_resolution[0] * grid_resolution[1], 2)
        grid_pos = torch.stack(torch.meshgrid(x_linspace, y_linspace, indexing="ij")).flatten(start_dim=1).T
        #
        if ctx is not None:
            assert "grid_pos" not in ctx
            ctx["grid_pos"] = grid_pos
        return grid_pos

    # noinspection PyUnusedLocal
    def getitem_mesh_edges(self, idx, ctx=None):
        assert self.grid_resolution is None
        if self.radius_graph_r is None:
            # radius graph is created on GPU
            return None
        sim_name = self._get_sim_name(idx)
        # load positions
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        if self.supernode_edge_mode == "mesh_to_mesh":
            # generate mesh_to_supernode edges by creating mesh_to_mesh edges and filtering them
            # this makes sure to include a self connection but leads to dataloading bottlenecks on slow CPUs
            # mesh to mesh interactions -> scales quadratically O(num_mesh_points^2)
            if self.num_supernodes is None:
                # normal flow direction
                flow = "source_to_target"
            else:
                # inverted flow direction is required to have sorted dst_indices
                flow = "target_to_source"
            edges = radius_graph(
                x=mesh_pos,
                r=self.radius_graph_r,
                max_num_neighbors=self.radius_graph_max_num_neighbors,
                loop=True,
                flow=flow,
            )
            if self.num_supernodes is not None:
                # reduce to mesh to supernodes interactions -> scales linearly O(num_supernodes * num_mesh_points)
                # NOTE: using radius(...) as a self-loop is required, which is tricky with radius(...)
                # this is because radius(...) can contain self-loops but if max_num_neighbors is set it doesnt always
                # contain them, so one would have to add the self loop depending on if it is already contained or not
                generator = self._get_generator(idx)
                perm = torch.randperm(len(mesh_pos), generator=generator)[:self.num_supernodes]
                is_supernode_edge = torch.isin(edges[0], perm)
                edges = edges[:, is_supernode_edge]
        elif self.supernode_edge_mode == "mesh_to_supernode":
            assert self.num_supernodes is not None
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
        else:
            raise NotImplementedError
        return edges.T

    def getitem_mesh_to_grid_edges(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert self.num_supernodes is None
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        grid_pos = self.getitem_grid_pos(idx, ctx=ctx)
        # create graph between mesh and regular grid points
        if self.radius_graph_r is None:
            # created on GPU
            return None
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
        assert self.num_supernodes is None
        grid_pos = self.getitem_grid_pos(idx, ctx=ctx)
        query_pos = self.getitem_query_pos(idx, ctx=ctx)
        # create graph between mesh and regular grid points
        if self.radius_graph_r is None:
            # created on GPU
            return None
        edges = radius(
            x=grid_pos,
            y=query_pos,
            r=self.radius_graph_r,
            max_num_neighbors=self.radius_graph_max_num_neighbors,
        ).T
        # edges is (num_points, 2)
        return edges

    def getshape_x(self):
        sim_name, timestep_to_fname = self.samples[0]
        num_channels = torch.load(self.source_root / sim_name / timestep_to_fname[0]).T.size(1)
        return None, num_channels * self.num_input_timesteps

    def getshape_target(self):
        sim_name, timestep_to_fname = self.samples[0]
        num_channels = torch.load(self.source_root / sim_name / timestep_to_fname[0]).T.size(1)
        return None, num_channels

    def getitem_target_t0(self, idx, ctx=None):
        assert self.num_input_timesteps == float("inf")
        # t0 is never returned in getitem_target
        sim_name, timestep_to_fname = self.samples[idx]
        data = torch.load(self.source_root / sim_name / timestep_to_fname[0]).T
        data = self._downsample_query(data, idx=idx, ctx=ctx)
        # data has shape (num_points, num_channels)
        data -= self.mean.view(1, -1)
        data /= self.std.view(1, -1)
        # data is sometimes stored as float16 for reduced storage requirements
        data = data.float()
        # clamp to remove outliers
        data = self._clamp(data)
        return data

    def getitem_target(self, idx, ctx=None):
        if self.num_input_timesteps == float("inf"):
            # targets for rollout -> return all but first timestep
            sim_name, timestep_to_fname = self.samples[idx]
            data = [
                torch.load(self.source_root / sim_name / timestep_to_fname[i]).T
                for i in range(1, self.max_timestep + 1)
            ]
            data = torch.stack([self._downsample_query(item, idx=idx, ctx=ctx) for item in data], dim=2)
            # data has shape (num_points, num_channels, max_timestep)
            data -= self.mean.view(1, -1, 1)
            data /= self.std.view(1, -1, 1)
        else:
            seqidx = idx // self.max_timestep
            sim_name, timestep_to_fname = self.samples[seqidx]
            # dataset for training -> predict random timestep
            timestep = self.getitem_timestep(idx, ctx=ctx)
            # target is the data from the next timestep
            data = torch.load(self.source_root / sim_name / timestep_to_fname[timestep + 1]).T
            # data has shape (num_points, num_channels)
            data = self._downsample_query(data, idx=idx, ctx=ctx)
            data -= self.mean.view(1, -1)
            data /= self.std.view(1, -1)
        # data is sometimes stored as float16 for reduced storage requirements
        data = data.float()
        # clamp to remove outliers
        data = self._clamp(data)
        return data

    def getitem_reconstruction_input(self, idx, ctx=None):
        # return target but with input downsampling
        assert self.num_input_timesteps != float("inf")
        seqidx = idx // self.max_timestep
        sim_name, timestep_to_fname = self.samples[seqidx]
        # dataset for training -> predict random timestep
        timestep = self.getitem_timestep(idx, ctx=ctx)
        # target is the data from the next timestep
        data = torch.load(self.source_root / sim_name / timestep_to_fname[timestep + 1]).T
        # data has shape (num_points, num_channels)
        data = self._downsample_input(data, idx=idx, ctx=ctx)
        data -= self.mean.view(1, -1)
        data /= self.std.view(1, -1)
        # data is sometimes stored as float16 for reduced storage requirements
        data = data.float()
        # clamp to remove outliers
        data = self._clamp(data)
        return data

    def getitem_reconstruction_output(self, idx, ctx=None):
        # return target for reconstruction loss
        assert self.num_input_timesteps != float("inf")
        seqidx = idx // self.max_timestep
        sim_name, timestep_to_fname = self.samples[seqidx]
        # dataset for training -> predict random timestep
        timestep = self.getitem_timestep(idx, ctx=ctx)
        # target is the data from the next timestep
        # TODO can be cached
        data = torch.load(self.source_root / sim_name / timestep_to_fname[timestep + 1]).T
        # data has shape (num_points, num_channels)
        data = self._downsample_reconstruction_output(data, idx=idx, ctx=ctx)
        data -= self.mean.view(1, -1)
        data /= self.std.view(1, -1)
        # data is sometimes stored as float16 for reduced storage requirements
        data = data.float()
        # clamp to remove outliers
        data = self._clamp(data)
        return data

    def _clamp(self, data):
        # clamp to remove outliers
        if self.clamp is not None:
            if self.clamp_mode == "hard":
                data = data.clamp(-self.clamp, self.clamp)
            elif self.clamp_mode == "log":
                # convert values larger than self.clamp to logscale
                apply = data.abs() > self.clamp
                values = data[apply]
                # TODO torch.log1p should be equivalent
                data[apply] = torch.sign(values) * (self.clamp + torch.log(1 + values.abs()) - np.log(1 + self.clamp))
            else:
                raise NotImplementedError
        return data

    def getitem_x(self, idx, ctx=None):
        if self.num_input_timesteps == float("inf"):
            # dataset for rollout -> return first timestep repeated num_input_timesteps times
            sim_name, timestep_to_fname = self.samples[idx]
            data = torch.load(self.source_root / sim_name / timestep_to_fname[0]).T
            # data has shape (num_points, num_channels)
            data = self._downsample_input(data, idx=idx, ctx=ctx)
            data -= self.mean.view(1, -1)
            data /= self.std.view(1, -1)
        else:
            # dataset for training -> predict random timestep
            seqidx = idx // self.max_timestep
            sim_name, timestep_to_fname = self.samples[seqidx]
            # get the timestep of the latest history [0, max_timesteps)
            timestep = self.getitem_timestep(idx, ctx=ctx)
            data = torch.stack([
                torch.load(self.source_root / sim_name / timestep_to_fname[max(0, i)]).T
                for i in range(timestep - self.num_input_timesteps + 1, timestep + 1)
            ], dim=1)
            # data has shape [num_points, num_input_timesteps, num_channels]
            data = self._downsample_input(data, idx=idx, ctx=ctx)
            data -= self.mean.view(1, 1, -1)
            data /= self.std.view(1, 1, -1)
            # flatten timesteps
            data = einops.rearrange(data, "num_points timesteps num_channels -> num_points (timesteps num_channels)")
        # data is sometimes stored as float16 for reduced storage requirements
        data = data.float()
        # clamp to remove outliers
        data = self._clamp(data)
        return data

    def getitem_interpolated(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert self.standardize_query_pos
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        # generate grid positions (these are different than getitem_gridpos because interpolate requires xy indexing)
        # it should be the same if indexing=ij since the mapping and inverse mapping consider the change in indexing
        # but for consistency with scipy.interpolate xy was chosen
        x_linspace = torch.linspace(0, self.max_x_pos, self.grid_resolution[1])
        y_linspace = torch.linspace(0, self.max_y_pos, self.grid_resolution[0])
        grid_pos = torch.meshgrid(x_linspace, y_linspace, indexing="xy")
        x = self.getitem_x(idx, ctx=ctx)
        grid = torch.from_numpy(
            scipy.interpolate.griddata(
                mesh_pos.unbind(1),
                x,
                grid_pos,
                method="linear",
                fill_value=0.,
            ),
        ).float()

        # check for correctness of interpolation
        # import matplotlib.pyplot as plt
        # import os
        # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        # plt.scatter(mesh_pos[:, 0], mesh_pos[:, 1], c=x[:, 4])
        # plt.show()
        # plt.clf()
        # plt.imshow(grid[:, :, 4], origin="lower")
        # plt.show()
        # plt.clf()
        # import torch.nn.functional as F
        # grid = einops.rearrange(grid, "h w dim -> 1 dim h w")
        # query_pos = self.getitem_query_pos(idx, ctx=ctx)
        # query_pos = einops.rearrange(query_pos, "num_points ndim -> 1 num_points 1 ndim")
        # mesh_values = F.grid_sample(input=grid, grid=query_pos, align_corners=False).squeeze(-1)
        # plt.scatter(*query_pos.squeeze().unbind(1), c=mesh_values[0, 4])
        # plt.show()
        # plt.clf()

        # grid has shape (height width dim) -> latent models expect dim-last format
        return grid
