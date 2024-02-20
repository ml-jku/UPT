import einops
import torch
from torch import nn

from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier


class RansGridSimpleCnn(SingleModelBase):
    def __init__(
            self,
            dim,
            depth=4,
            num_poolings=3,
            pooling_kernel_sizes=None,
            num_groups=8,
            add_type_token=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_poolings = num_poolings
        self.pooling_kernel_sizes = pooling_kernel_sizes
        self.num_groups = num_groups
        self.add_type_token = add_type_token
        self.resolution = self.data_container.get_dataset().grid_resolution
        self.ndim = len(self.resolution)
        assert num_poolings <= depth - 1
        assert dim % 2 ** (depth - 1) == 0
        assert all(self.resolution[0] == resolution for resolution in self.resolution[1:])
        assert self.resolution[0] % (2 ** (depth - 1)) == 0
        dim_per_block = [dim // (2 ** (depth - i - 1)) for i in range(depth)]

        # sdf + grid_pos
        if self.data_container.get_dataset().concat_pos_to_sdf:
            input_dim = 4
        else:
            input_dim = 1

        # down path of latent.unet_gino_model
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                # first block has no pooling and goes from input_dim -> dim_per_block[0] -> dim_per_block[1]
                block = nn.Sequential(
                    # pooling
                    nn.Identity(),
                    # conv1
                    nn.GroupNorm(num_groups=1, num_channels=input_dim),
                    nn.Conv3d(input_dim, dim_per_block[0] // 2, kernel_size=3, padding=1, bias=False),
                    nn.GELU(),
                    # conv2
                    nn.GroupNorm(num_groups=num_groups, num_channels=dim_per_block[0] // 2),
                    nn.Conv3d(dim_per_block[0] // 2, dim_per_block[0], kernel_size=3, padding=1, bias=False),
                    nn.GELU(),
                )
            else:
                if depth - num_poolings <= i:
                    if pooling_kernel_sizes is None:
                        kernel_size = 2
                    else:
                        kernel_size = pooling_kernel_sizes[i - (depth - num_poolings)]
                    pooling = nn.MaxPool3d(kernel_size=kernel_size, stride=kernel_size)
                else:
                    pooling = nn.Identity()
                block = nn.Sequential(
                    # pooling
                    pooling,
                    # conv1
                    nn.GroupNorm(num_groups=num_groups, num_channels=dim_per_block[i] // 2),
                    nn.Conv3d(dim_per_block[i] // 2, dim_per_block[i] // 2, kernel_size=3, padding=1, bias=False),
                    nn.GELU(),
                    # conv2
                    nn.GroupNorm(num_groups=num_groups, num_channels=dim_per_block[i] // 2),
                    nn.Conv3d(dim_per_block[i] // 2, dim_per_block[i], kernel_size=3, padding=1, bias=False),
                    nn.GELU(),
                )
            self.blocks.append(block)

        if add_type_token:
            self.type_token = nn.Parameter(torch.empty(size=(1, 1, dim,)))
        else:
            self.type_token = None

        self.static_ctx["grid_resolution"] = self.resolution
        self.static_ctx["ndim"] = self.ndim
        self.output_shape = (int(self.resolution[0] // (2 ** (depth - 1)) ** self.ndim), dim)

    def model_specific_initialization(self):
        if self.add_type_token:
            nn.init.trunc_normal_(self.type_token)

    def get_model_specific_param_group_modifiers(self):
        modifiers = []
        if self.add_type_token:
            modifiers += [ExcludeFromWdByNameModifier(name="type_token")]
        return modifiers

    def forward(self, x):
        # sdf is passed as dim-last with spatial -> convert to dim-first with spatial
        x = einops.rearrange(x, "batch_size height width depth dim -> batch_size dim height width depth")
        # embed
        for block in self.blocks:
            x = block(x)
        # flatten to tokens
        x = einops.rearrange(x, "batch_size dim height width depth -> batch_size (height width depth) dim")
        if self.add_type_token:
            x = x + self.type_token
        return x
