import torch
from torchvision.transforms.functional import to_pil_image

from .base.dataset_base import DatasetBase


class DummyDataset(DatasetBase):
    def __init__(
            self,
            x_shape,
            size=None,
            n_classes=10,
            n_abspos=10,
            is_multilabel=False,
            to_image=False,
            semi_percent=None,
            num_timesteps=10,
            force_timestep_zero=False,
            mode="on-the-fly",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size
        self.x_shape = x_shape
        self._n_classes = n_classes
        self.n_abspos = n_abspos
        self._is_multilabel = is_multilabel
        self.to_image = to_image
        self.semi_percent = semi_percent
        self.num_timesteps = num_timesteps
        self.force_timestep_zero = force_timestep_zero
        self.mode = mode
        assert semi_percent is None or 0. <= semi_percent <= 1.
        assert mode in ["on-the-fly", "preloaded"]
        if self.mode == "preloaded":
            self.x = torch.randn(len(self), *self.x_shape, generator=torch.Generator().manual_seed(0))
            self.y = torch.randint(
                low=0,
                high=max(2, self.getdim_class()),
                size=(1,),
                generator=torch.Generator().manual_seed(0),
            ).tolist()
        else:
            self.x = None
            self.y = None

    def __len__(self):
        # return a large value divisible by 2 to avoid specifying a size when the dataset is only used
        # for the eval_trainer to know the input shapes
        return self.size or 131072

    def getitem_x(self, idx, ctx=None):
        if self.x is not None:
            return self.x[idx]
        x = torch.randn(*self.x_shape, generator=torch.Generator().manual_seed(int(idx)))
        if self.to_image:
            x = to_pil_image(x)
        return x

    # noinspection PyUnusedLocal
    def getitem_timestep(self, idx, ctx=None):
        if self.force_timestep_zero:
            return 0
        max_timestep = self.num_timesteps - self.x_shape[0]
        timestep = torch.randint(max_timestep, size=(1,), generator=torch.Generator().manual_seed(int(idx)))
        return timestep

    def getshape_timestep(self):
        return self.num_timesteps,

    def getshape_class(self):
        return (self._n_classes,) if self._n_classes > 2 else (1,)

    def getitem_class(self, idx, ctx=None):
        if self.semi_percent is not None and (idx / len(self)) < self.semi_percent:
            return -1
        return self.getitem_class_all(idx, ctx=ctx)

    # noinspection PyUnusedLocal
    def getitem_class_all(self, idx, ctx=None):
        if self.y is not None:
            return self.y[idx]
        return torch.randint(
            low=0,
            high=max(2, self.getdim_class()),
            size=(1,),
            generator=torch.Generator().manual_seed(int(idx)),
        ).item()

    def getall_class(self):
        return [self.getitem_class(i) for i in range(len(self))]

    def getshape_abspos(self):
        return self.n_abspos,

    # noinspection PyUnusedLocal
    def getitem_abspos(self, idx, ctx=None):
        return torch.randint(
            low=0,
            high=self.n_abspos,
            size=(1,),
            generator=torch.Generator().manual_seed(idx),
        ).item()

    # noinspection PyUnusedLocal
    def getitem_semseg(self, idx, ctx=None):
        assert len(self.x_shape) == 3
        return torch.randint(0, 10, size=self.x_shape[1:], generator=torch.Generator().manual_seed(int(idx)))

    @staticmethod
    def getshape_semseg():
        return 10,

    @property
    def is_multilabel(self):
        return self._is_multilabel
