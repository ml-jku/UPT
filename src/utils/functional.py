import einops
import numpy as np
import torch
import torch.nn.functional as F


def get_powers_of_two(min_value, max_value):
    powers_of_two = []
    if max_value > 0:
        powers_of_two += [2 ** i for i in range(int(np.log2(max_value)) + 1)]
    return [p for p in powers_of_two if p >= min_value]


def is_power_of_two(value):
    return np.log2(value).is_integer()


def image_to_pyramid(x, num_scales):
    scaled_imgs = [x]
    _, _, height, width = x.shape
    for _ in range(num_scales - 1):
        height //= 2
        width //= 2
        # interpolate is not supported in bfloat16
        with torch.autocast(device_type=str(x.device).split(":")[0], enabled=False):
            scaled = F.interpolate(x, size=[height, width], mode="bilinear", align_corners=True)
        scaled_imgs.append(scaled)
    return scaled_imgs


def gram_matrix(x):
    _, c, h, w = x.shape
    x = einops.rearrange(x, "b c h w -> b c (h w)")
    xt = einops.rearrange(x, "b c hw -> b hw c")
    gram = torch.bmm(x, xt) / (c * h * w)
    return gram

def to_ndim(x, ndim):
    return x.reshape(*x.shape, *(1,) * (ndim - x.ndim))
