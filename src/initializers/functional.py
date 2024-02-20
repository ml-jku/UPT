import torch.nn as nn

ALL_BATCHNORMS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LazyBatchNorm1d,
    nn.LazyBatchNorm2d,
    nn.LazyBatchNorm3d,
    nn.SyncBatchNorm,
)

_ALL_NORMS = (
    *ALL_BATCHNORMS,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.GroupNorm,
    nn.LocalResponseNorm,
)


def initialize_norms_as_noaffine(m):
    if isinstance(m, _ALL_NORMS):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.)


def initialize_norms_as_identity(m):
    if isinstance(m, _ALL_NORMS):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        if m.weight is not None:
            nn.init.constant_(m.weight, 0.)


def initialize_layernorm_as_noaffine(m):
    if isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.)


def initialize_layernorm_as_identity(m):
    if isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        if m.weight is not None:
            nn.init.constant_(m.weight, 0.)
    else:
        raise NotImplementedError


def initialize_batchnorm_as_noaffine(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.)


def initialize_batchnorm_as_identity(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        if m.weight is not None:
            nn.init.constant_(m.weight, 0.)
    else:
        raise NotImplementedError


def initialize_linear_bias_to_zero(m):
    if isinstance(m, nn.Linear):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)

def initialize_conv_bias_to_zero(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)

def initialize_xavier_uniform_zero_bias(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)

def initialize_qkv_seperately(model):
    # https://github.com/facebookresearch/moco-v3/blob/main/vits.py#L35
    for full_name, module in model.named_modules():
        last_name = full_name.split(".")[-1]
        if last_name == "qkv":
            # treat the weights of Q, K, V separately
            val = (6 / (module.weight.shape[0] // 3 + module.weight.shape[1])) ** 0.5
            nn.init.uniform_(module.weight, -val, val)
        if last_name == "qkv_mlpin":
            # treat the weights of Q, K, V and MLP-in separately
            # only implemented for mlp_ratio=4
            input_dim = module.weight.shape[1]
            assert module.weight.shape[0] == 7 * input_dim
            qkv_bound = (3 / input_dim) ** 0.5
            mlpin_bound = (6 / (5 * input_dim)) ** 0.5
            nn.init.uniform_(module.weight[:3 * input_dim], -qkv_bound, qkv_bound)
            nn.init.uniform_(module.weight[3 * input_dim:], -mlpin_bound, mlpin_bound)

def initialize_modulation_seperately(model):
    for full_name, module in model.named_modules():
        last_name = full_name.split(".")[-1]
        if last_name == "modulation":
            # a modulation produces a stack of vectors -> treat each vector seperately
            val = (6 / (module.weight.shape[0] // 2 + module.weight.shape[1])) ** 0.5
            nn.init.uniform_(module.weight, -val, val)

def initialize_seperately(model, name, denominator):
    for full_name, module in model.named_modules():
        last_name = full_name.split(".")[-1]
        if last_name == name:
            val = (6 / (module.weight.shape[0] // denominator + module.weight.shape[1])) ** 0.5
            nn.init.uniform_(module.weight, -val, val)
