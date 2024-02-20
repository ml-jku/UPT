from .base.freezer_base import FreezerBase


class FullFreezer(FreezerBase):
    def __str__(self):
        return type(self).__name__

    def _update_state(self, model, requires_grad):
        model.eval()
        for param in model.parameters():
            param.requires_grad = requires_grad
