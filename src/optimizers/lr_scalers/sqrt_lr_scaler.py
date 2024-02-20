import math


class SqrtLrScaler:
    def __init__(self, divisor=256):
        super().__init__()
        self.divisor = divisor

    def __str__(self):
        return f"{type(self).__name__}(divisor={self.divisor})"

    def scale_lr(self, base_lr, lr_scale_factor):
        return base_lr * math.sqrt(lr_scale_factor / self.divisor)
