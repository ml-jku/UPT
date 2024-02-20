import os
import json
import h5py
import wget
import zipfile

import torch
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph, radius
from torch_geometric.transforms import KNNGraph
from .base.dataset_base import DatasetBase

from kappadata.copying.image_folder import copy_imagefolder_from_global_to_local
from distributed.config import barrier, is_data_rank0
import einops

URLS = {
    "tgv2d": "https://zenodo.org/records/10021926/files/2D_TGV_2500_10kevery100.zip",
    "rpf2d": "https://zenodo.org/records/10021926/files/2D_RPF_3200_20kevery100.zip",
    "ldc2d": "https://zenodo.org/records/10021926/files/2D_LDC_2708_10kevery100.zip",
    "dam2d": "https://zenodo.org/records/10021926/files/2D_DAM_5740_20kevery100.zip",
    "tgv3d": "https://zenodo.org/records/10021926/files/3D_TGV_8000_10kevery100.zip",
    "rpf3d": "https://zenodo.org/records/10021926/files/3D_RPF_8000_10kevery100.zip",
    "ldc3d": "https://zenodo.org/records/10021926/files/3D_LDC_8160_10kevery100.zip",
}

class LagrangianDataset(DatasetBase):
    def __init__(
            self,
            name,
            n_input_timesteps=6,
            n_pushforward_timesteps=0,
            graph_mode='radius_graph',
            knn_graph_k=1,
            radius_graph_r=0.05,
            radius_graph_max_num_neighbors=int(1e10),
            split="train",
            test_mode='parts_traj',
            n_supernodes=None,
            num_points_range=None,
            global_root=None,
            local_root=None,
            seed=None,
            pos_scale=200,
            **kwargs,
    ):
        super().__init__(**kwargs)
        assert name in URLS.keys(), f"Dataset {name} not available."
        assert split in ["train", "valid", "test"], f"Split {split} not available."
        assert n_input_timesteps > 1, f"num_inputs_timesteps must be greater than 1 to calculate input velocities."
        assert graph_mode in ['knn', 'radius_graph', 'radius_graph_with_supernodes'], f"graph_mode {graph_mode} not available."
        assert test_mode in ['full_traj', 'parts_traj']
        self.n_input_timesteps = n_input_timesteps
        self.n_pushforward_timesteps = n_pushforward_timesteps
        self.knn_graph_k = knn_graph_k
        self.radius_graph_r = radius_graph_r
        self.radius_graph_max_num_neighbors = radius_graph_max_num_neighbors
        self.graph_mode = graph_mode
        self.split = split
        self.num_points_range = num_points_range
        self.test_mode = test_mode
        self.seed = seed
        # TODO: Rewrite this for global and local
        ds_name = os.path.splitext(URLS[name].split('/')[-1])[0]
        global_root, local_root = self._get_roots(global_root, local_root, "lagrangian_dataset")
        # TODO: Implement local dataloading
        # load data from global_root
        self.source_root = global_root / ds_name
        # Check if dataset needs to be downloaded
        if not os.path.isdir(self.source_root):
            self.logger.info(f"downloading '{ds_name}'")
            self.download(name, global_root)
        self.logger.info(f"data_source (global): '{self.source_root}'")
        assert self.source_root.exists(), f"'{self.source_root.as_posix()}' doesn't exist"

        self.data = self.load_dataset(self.source_root, split)  
        self.metadata = self.load_metadata(self.source_root)
        self.traj_keys = list(self.data.keys())
        self.n_traj = len(self.traj_keys)

        # Get number of particles
        self.n_particles = self.metadata['num_particles_max']
        if n_supernodes:
            self.n_supernodes = n_supernodes
        else:
            self.n_supernodes = self.n_particles

        # Normalization stats
        self.vel_mean = torch.tensor(self.metadata['vel_mean'])
        self.vel_std = torch.tensor(self.metadata['vel_std'])
        self.acc_mean = torch.tensor(self.metadata['acc_mean'])
        self.acc_std = torch.tensor(self.metadata['acc_std'])

        # Check for PBC
        if any(self.metadata['periodic_boundary_conditions']):
            bounds = torch.tensor(self.metadata['bounds'])
            self.box = bounds[:, 1] - bounds[:, 0]
            # Scaling for positional embedding
            # Positional embedding function is the same as for the ViT
            # The range is from 0-197 -> we use as a max 200
            self.pos_offset = bounds[:,0]
            self.pos_scale = pos_scale / self.box
        else:
            assert NotImplementedError

        if self.split == "train":
            self.n_seq = self.metadata['sequence_length_train']
            # num_input_timesteps is needed to train
            # one additional position is needed as target (in the form of velocity or acceleration)
            # num_pushforward_timesteps is the difference of timesteps between the input sequence and the output sequence
            # Number of samples per trajectory is therefore:
            self.n_per_traj = self.n_seq - self.n_input_timesteps - self.n_pushforward_timesteps
            self.getter = self.get_window
        else:
            self.n_seq = self.metadata['sequence_length_test']
            if self.test_mode == 'full_traj':
                self.getter = self.get_full_trajectory
                self.n_per_traj = 1
            else:
                self.n_sub = n_input_timesteps + n_pushforward_timesteps
                self.n_per_traj = self.n_seq // self.n_sub
                self.getter = self.get_trajectory
        
    def load_dataset(self, path, split):
        # Load dataset
        data = h5py.File(os.path.join(path, split + '.h5'))
        return data
    
    def load_metadata(self, path):
        # Load metadata
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.loads(f.read())
        return metadata

    def download(self, name, path):
        url = URLS[name]
        filename = os.path.split(url)[-1]
        filepath = wget.download(url, out=os.path.join(path, filename))
        # unzip the dataset
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(path)
        # remove the zip file
        os.remove(filepath)

    def get_window(self, idx: int, ctx=None, downsample=False):
        assert ctx is not None
        if "window" in ctx:
            if downsample and "perm" in ctx:
                perm = ctx["perm"]
                return ctx["window"][0][:,perm,:], ctx["window"][1][perm]
            else:
                return ctx["window"]
        # Trajectory index
        i_traj = idx // self.n_per_traj
        traj = self.data[self.traj_keys[i_traj]]
        # Index where to start in traj
        start_idx = idx % self.n_per_traj
        end_idx = start_idx+self.n_input_timesteps+self.n_pushforward_timesteps+1
        ctx['time_idx'] = torch.arange(start_idx, end_idx)
        ctx['traj_idx'] = i_traj
        positions = traj['position'][ctx['time_idx']]
        particle_types = traj['particle_type']
        positions = torch.tensor(positions)
        particle_types = torch.tensor(particle_types)
        # Subsampling
        if self.num_points_range:
            if self.num_points_range[0] == self.num_points_range[1]:
                # fixed num_points_range
                end = self.num_points_range[1]
            else:
                lb, ub = self.num_points_range
                num_points_range = torch.rand(size=(1,), generator=None).item() * (ub - lb) + lb
                end = int(num_points_range)
            # uniform sampling
            perm = torch.randperm(self.n_particles, generator=None)[:end]
            ctx["perm"] = perm
        # window has the full data without subsampling in, 
        # so in one sampling we can access both the downsampled and the full version
        ctx["window"] = positions, particle_types
        # Save the maximum number of particles to the ctx
        ctx['max_particles'] = self.n_particles
        if downsample and "perm" in ctx:
            return ctx["window"][0][:,perm,:], ctx["window"][1][perm]
        else:
            return ctx["window"]

    def get_trajectory(self, idx: int, ctx=None, downsample=False):
        # Trajectory index
        i_traj = idx // self.n_per_traj
        traj = self.data[self.traj_keys[i_traj]]
        # Index where to start in traj
        start_idx = (idx % self.n_per_traj) * self.n_sub
        end_idx = start_idx + self.n_sub
        ctx['time_idx'] = torch.arange(start_idx, end_idx)
        ctx['traj_idx'] = i_traj
        positions = traj['position'][ctx['time_idx']]
        particle_types = traj['particle_type']
        return torch.tensor(positions), torch.tensor(particle_types)
    
    def get_full_trajectory(self, idx: int, ctx=None, downsample=False):
        # Trajectory index
        i_traj = idx
        traj = self.data[self.traj_keys[i_traj]]
        ctx['time_idx'] = torch.arange(0, len(traj['position']))
        ctx['traj_idx'] = i_traj
        positions = traj['position'][ctx['time_idx']]
        particle_types = traj['particle_type']
        return torch.tensor(positions), torch.tensor(particle_types)

    def get_velocities(self, positions):
        velocities = positions[1:,:,:] - positions[:-1,:,:]
        if self.box is not None:
            # Calculation of PBC is done like in jax_md.space.periodic
            velocities = (velocities + self.box * 0.5) % self.box - 0.5 * self.box
        # Normalization
        velocities = self.normalize_vel(velocities)
        return velocities
    
    def get_accelerations(self, positions):
        next_velocities = positions[2:,:,:] - positions[1:-1,:,:]
        current_velocities = positions[1:-1,:,:] - positions[0:-2,:,:]
        if self.box is not None:
            # Calculation of PBC is done like in jax_md.space.periodic
            next_velocities = (next_velocities + self.box * 0.5) % self.box - 0.5 * self.box
            current_velocities = (current_velocities + self.box * 0.5) % self.box - 0.5 * self.box
        accelerations = next_velocities - current_velocities
        # Normalization
        accelerations = self.normalize_acc(accelerations)
        return accelerations

    def __len__(self):
        return self.n_traj * self.n_per_traj
    
    def __getitem__(self, idx):
        if self.split == "train":
            positions, particle_types = self.get_window(idx)
        else:
            positions, particle_types = self.get_trajectory(idx)
        return positions, particle_types
    
    def scale_pos(self, pos):
        pos = pos - self.pos_offset.to(pos.device)
        pos = pos * self.pos_scale.to(pos.device)
        return pos
    
    def unscale_pos(self, pos):
        pos = pos / self.pos_scale.to(pos.device)
        pos = pos + self.pos_offset.to(pos.device)
        return pos
    
    def normalize_vel(self, vel):
        vel = vel - self.vel_mean.to(vel.device)
        vel = vel / self.vel_std.to(vel.device)
        return vel
    
    def unnormalize_vel(self, vel):
        vel = vel * self.vel_std.to(vel.device)
        vel = vel + self.vel_mean.to(vel.device)
        return vel
    
    def normalize_acc(self, acc):
        acc = acc - self.acc_mean.to(acc.device)
        acc = acc / self.acc_std.to(acc.device)
        return acc
    
    def unnormalize_acc(self, acc):
        acc = acc * self.acc_std.to(acc.device)
        acc = acc + self.acc_mean.to(acc.device)
        return acc

    def _get_generator(self, idx):
        if self.split == "valid" or self.split == "test":
            assert self.seed is not None
        if self.seed is not None:
            return torch.Generator().manual_seed(int(idx) + self.seed)
        return None

    # Kappadata getters
    def getitem_timestep(self, idx, ctx=None):
        if self.split == 'test' or self.split == 'valid':
            if self.test_mode == 'full_traj':
                return self.n_input_timesteps - 1
            else:
                return (idx % self.n_per_traj) * self.n_sub + self.n_input_timesteps - 1
        return idx % self.n_per_traj + self.n_input_timesteps - 1
    
    def getshape_timestep(self):
        return max(self.metadata['sequence_length_train'], self.metadata['sequence_length_test']),
    
    def getitem_curr_pos(self, idx, ctx=None):
        positions, _ = self.getter(idx, ctx, downsample=True)
        input_positions = positions[:self.n_input_timesteps,:,:]
        current_input_position = input_positions[-1,:,:]
        # Scale position so it fits for the positional embedding
        current_input_position = self.scale_pos(current_input_position)
        return current_input_position
    
    def getitem_curr_pos_full(self, idx, ctx=None):
        positions, _ = self.getter(idx, ctx, downsample=False)
        input_positions = positions[:self.n_input_timesteps,:,:]
        current_input_position = input_positions[-1,:,:]
        # Scale position so it fits for the positional embedding
        current_input_position = self.scale_pos(current_input_position)
        return current_input_position
    
    def getitem_x(self, idx, ctx=None):
        positions, _ = self.getter(idx, ctx, downsample=True)
        # Calculate velocities
        input_features = self.get_velocities(positions[:self.n_input_timesteps,:,:])
        # Reshape for mesh_collator (num_input_timesteps, num_channels, num_points)
        input_features = input_features.permute(0,2,1)
        return input_features
    
    def getitem_edge_index(self, idx, ctx=None, downsample=True):
        positions, _ = self.getter(idx, ctx, downsample=True)
        input_positions = positions[:self.n_input_timesteps,:,:]
        current_input_position = input_positions[-1,:,:]
        if self.graph_mode == 'knn':
            knn = KNNGraph(k=self.knn_graph_k, loop=True, force_undirected=True)
            edge_index = knn(Data(pos=current_input_position)).edge_index.T
        elif self.graph_mode == 'radius_graph':
            edge_index = radius_graph(x=current_input_position, 
                                      r=self.radius_graph_r, 
                                      max_num_neighbors=self.radius_graph_max_num_neighbors, 
                                      loop=True).T
        elif self.graph_mode == 'radius_graph_with_supernodes':
            # select supernodes
            generator = self._get_generator(idx)
            perm_supernodes = torch.randperm(current_input_position.shape[0], generator=generator)[:self.n_supernodes]
            supernodes_pos = current_input_position[perm_supernodes]
            # create edges: this can include self-loop or not depending on how many neighbors are found.
            # if too many neighbors are found, neighbors are selected randomly which can discard the self-loop
            edge_index = radius(
                x=current_input_position,
                y=supernodes_pos,
                r=self.radius_graph_r,
                max_num_neighbors=self.radius_graph_max_num_neighbors,
            )
            # correct supernode index
            edge_index[0] = perm_supernodes[edge_index[0]]
            edge_index = edge_index.T
        return edge_index
    
    def getitem_edge_index_target(self, idx, ctx=None, downsample=True):
        positions, _ = self.getter(idx, ctx, downsample=True)
        target_position = positions[-1,:,:]
        if self.graph_mode == 'knn':
            knn = KNNGraph(k=self.knn_graph_k, loop=True, force_undirected=True)
            edge_index = knn(Data(pos=target_position)).edge_index.T
        elif self.graph_mode == 'radius_graph':
            edge_index = radius_graph(x=target_position, 
                                      r=self.radius_graph_r, 
                                      max_num_neighbors=self.radius_graph_max_num_neighbors, 
                                      loop=True).T
        elif self.graph_mode == 'radius_graph_with_supernodes':
            # select supernodes
            generator = self._get_generator(idx)
            perm_supernodes = torch.randperm(target_position.shape[0], generator=generator)[:self.n_supernodes]
            supernodes_pos = target_position[perm_supernodes]
            # create edges: this can include self-loop or not depending on how many neighbors are found.
            # if too many neighbors are found, neighbors are selected randomly which can discard the self-loop
            edge_index = radius(
                x=target_position,
                y=supernodes_pos,
                r=self.radius_graph_r,
                max_num_neighbors=self.radius_graph_max_num_neighbors,
            )
            # correct supernode index
            edge_index[0] = perm_supernodes[edge_index[0]]
            edge_index = edge_index.T
        return edge_index
    
    # Only used in GNS
    def getitem_edge_features(self, idx, ctx=None, downsample=True):
        edge_index = self.getitem_edge_index_target(idx, ctx, downsample=downsample)
        positions, _ = self.getter(idx, ctx, downsample=True)
        target_position = positions[-1,:,:]
        relative_displacement = target_position[edge_index[:,0]] - target_position[edge_index[:,1]]
        distance = ((target_position[edge_index[:,0]] - target_position[edge_index[:,1]])).norm(dim=-1)
        return torch.concat([relative_displacement, distance.unsqueeze(dim=-1)], dim=1)
    
    def getitem_target_vel(self, idx, ctx=None):
        positions, _ = self.getter(idx, ctx, downsample=False)
        target_features = self.get_velocities(positions[-2:,:,:]).squeeze()
        return target_features
    
    def getitem_perm(self, idx, ctx=None):
        if ctx and 'perm' in ctx:
            return ctx['perm'], self.n_particles
        else:
            return torch.arange(self.n_particles), self.n_particles
    
    def getitem_target_vel_large_t(self, idx, ctx=None):
        positions, _ = self.getter(idx, ctx, downsample=False)
        target_features = self.get_velocities(positions[-3:,:,:]).squeeze()
        target_features = einops.rearrange(
            target_features,
            "num_input_timesteps n_particles dim -> n_particles (num_input_timesteps dim)"
        )
        return target_features
    
    def getitem_target_acc(self, idx, ctx=None):
        positions, _ = self.getter(idx, ctx, downsample=False)
        target_features = self.get_accelerations(positions[-3:,:,:]).squeeze()
        return target_features
    
    def getitem_target_pos(self, idx, ctx=None):
        positions, _ = self.getter(idx, ctx, downsample=False)
        if self.split == 'test' or self.split == 'valid':
            current_output_position = positions[self.n_input_timesteps:,:,:]
        else:
            current_output_position = positions[-1,:,:]
            current_output_position = self.scale_pos(current_output_position)
        return current_output_position
    
    def getitem_target_pos_encode(self, idx, ctx=None):
        positions, _ = self.getter(idx, ctx, downsample=True)
        if self.split == 'test' or self.split == 'valid':
            current_output_position = positions[self.n_input_timesteps:,:,:]
        else:
            current_output_position = positions[-1,:,:]
            current_output_position = self.scale_pos(current_output_position)
        return current_output_position
    
    def getitem_last_but_one_pos(self, idx, ctx=None):
        positions, _ = self.getter(idx, ctx, downsample=False)
        current_output_position = positions[-2,:,:]
        current_output_position = self.scale_pos(current_output_position)
        return current_output_position
    
    def getitem_particle_type(self, idx, ctx=None):
        _, particle_type = self.getter(idx, ctx)
        return particle_type
    
    def getitem_prev_acc(self, idx, ctx=None):
        positions, _ = self.getter(idx, ctx, downsample=False)
        prev_acc = self.get_accelerations(positions[-4:,:,:]).squeeze()
        return prev_acc[0,:,:]
    
    def getitem_prev_pos(self, idx, ctx=None):
        positions, _ = self.getter(idx, ctx, downsample=False)
        input_positions = positions[:self.n_input_timesteps,:,:]
        current_input_position = input_positions[-2,:,:]
        # Scale position so it fits for the positional embedding
        current_input_position = self.scale_pos(current_input_position)
        return current_input_position
    
    # Methods for rollout with large delta T
    def getitem_all_pos(self, idx, ctx=None):
        positions, _ = self.getter(idx, ctx, downsample=False)
        positions = self.scale_pos(positions)
        return positions
    
    def getitem_all_vel(self, idx, ctx=None):
        positions, _ = self.getter(idx, ctx, downsample=False)
        velocities = self.get_velocities(positions)
        return velocities

    