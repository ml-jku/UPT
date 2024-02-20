import torch


class ConcatFinalizer:
    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, features):
        if self.dim is None:
            assert len(features) == 1
            return features[0]
        return torch.concat(features, dim=self.dim)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__
