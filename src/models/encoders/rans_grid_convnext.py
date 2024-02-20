import numpy as np
import einops
import torch
from kappamodules.convolution import ConvNext
from torch import nn
import torch.nn.functional as F
from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier


class RansGridConvnext(SingleModelBase):
    def __init__(
            self,
            patch_size,
            dims,
            depths,
            kernel_size=7,
            depthwise=True,
            global_response_norm=True,
            drop_path_rate=0.,
            drop_path_decay=False,
            add_pos_tokens=False,
            upsample_size=None,
            upsample_mode="nearest",
            resolution=None,
            concat_pos_to_sdf=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.dims = dims
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.add_pos_tokens = add_pos_tokens
        self.upsample_size = upsample_size
        self.upsample_mode = upsample_mode
        self.resolution = resolution or self.data_container.get_dataset().grid_resolution
        self.ndim = len(self.resolution)

        # sdf + grid_pos
        concat_pos_to_sdf = concat_pos_to_sdf or self.data_container.get_dataset().concat_pos_to_sdf
        if concat_pos_to_sdf:
            input_dim = 4
        else:
            input_dim = 1


        self.model = ConvNext(
            patch_size=patch_size,
            input_dim=input_dim,
            dims=dims,
            depths=depths,
            ndim=self.ndim,
            drop_path_rate=drop_path_rate,
            drop_path_decay=drop_path_decay,
            kernel_size=kernel_size,
            depthwise=depthwise,
            global_response_norm=global_response_norm,
        )


        out_resolution = [r // 2 ** (len(depths) - 1) // patch_size for r in self.resolution]
        num_output_tokens = int(np.prod(out_resolution))
        if add_pos_tokens:
            self.pos_tokens = nn.Parameter(torch.empty(size=(1, num_output_tokens, dims[-1])))
        else:
            self.pos_tokens = None
        self.type_token = nn.Parameter(torch.empty(size=(1, 1, dims[-1])))

        self.static_ctx["grid_resolution"] = self.resolution
        self.static_ctx["ndim"] = self.ndim
        self.output_shape = (num_output_tokens, dims[-1])

    def model_specific_initialization(self):
        if self.add_pos_tokens:
            nn.init.trunc_normal_(self.pos_tokens)
        nn.init.trunc_normal_(self.type_token)

    def get_model_specific_param_group_modifiers(self):
        modifiers = [ExcludeFromWdByNameModifier(name="type_token")]
        if self.add_pos_tokens:
            modifiers += [ExcludeFromWdByNameModifier(name="pos_tokens")]
        return modifiers

    def forward(self, x):
        # sdf is passed as dim-last with spatial -> convert to dim-first with spatial
        x = einops.rearrange(x, "batch_size height width depth dim -> batch_size dim height width depth")
        # upsample
        if self.upsample_size is not None:
            if self.upsample_mode == "nearest":
                x = F.interpolate(x, size=self.upsample_size, mode=self.upsample_mode)
            else:
                x = F.interpolate(x, size=self.upsample_size, mode=self.upsample_mode, align_corners=True)

        # embed
        x = self.model(x)
        # flatten to tokens
        x = einops.rearrange(x, "batch_size dim height width depth -> batch_size (height width depth) dim")
        x = x + self.type_token
        if self.add_pos_tokens:
            x = x + self.pos_tokens.expand(len(x), -1, -1)
        return x
