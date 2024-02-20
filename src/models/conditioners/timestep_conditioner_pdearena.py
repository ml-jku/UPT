from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen
from kappamodules.init import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn

from models.base.single_model_base import SingleModelBase


class TimestepConditionerPdearena(SingleModelBase):
    def __init__(self, dim, cond_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.num_total_timesteps = self.data_container.get_dataset().getdim_timestep()
        self.dim = dim
        self.cond_dim = cond_dim or dim * 4
        self.static_ctx["condition_dim"] = self.cond_dim
        # buffer/modules
        self.register_buffer(
            "timestep_embed",
            get_sincos_1d_from_seqlen(seqlen=self.num_total_timesteps, dim=dim),
        )
        self.timestep_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, self.cond_dim),
            nn.GELU(),
        )
        # init
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_xavier_uniform_zero_bias)

    def forward(self, timestep, velocity):
        # checks + preprocess
        assert timestep.numel() == len(timestep)
        assert velocity.numel() == len(velocity)
        timestep = timestep.flatten()
        velocity = velocity.view(-1, 1).float()
        # for rollout timestep is simply initialized as 0 -> repeat to batch dimension
        if timestep.numel() == 1:
            timestep = timestep.repeat(velocity.numel())
        # embed
        timestep_embed = self.timestep_mlp(self.timestep_embed[timestep])
        return timestep_embed
