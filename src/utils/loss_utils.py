import torch


def apply_reduction(tensor, reduction="mean"):
    if tensor.dtype == torch.bool:
        tensor = tensor.float()
    if reduction == "mean":
        return tensor.mean()
    if reduction == "mean_per_sample":
        if tensor.ndim > 1:
            return tensor.flatten(start_dim=1).mean(dim=1)
        return tensor
    if reduction is None or reduction == "none":
        return tensor
    raise NotImplementedError
