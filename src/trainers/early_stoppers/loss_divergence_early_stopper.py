from collections import deque

import numpy as np

from callbacks.base.callback_base import CallbackBase
from .base.early_stopper_base import EarlyStopperBase


class LossDivergenceEarlyStopper(EarlyStopperBase):
    """
    stop training if loss diverges
    organize losses into [...] + [reference_window] + [tolerance_window]
    if avg(tolerance_window) > avg(reference_window) * tolerance_factor -> stop
    """

    def __init__(self, reference_window, tolerance_window, tolerance_factor, **kwargs):
        super().__init__(**kwargs)
        assert reference_window is not None and 1 <= reference_window, f"reference_window < 1 ({reference_window})"
        assert tolerance_window is not None and 1 <= tolerance_window, f"tolerance_window < 1 ({tolerance_window})"
        assert tolerance_factor is not None and 1. <= tolerance_factor, f"tolerance_factor < 1 ({tolerance_factor})"
        self.reference_window = reference_window
        self.tolerance_window = tolerance_window
        self.tolerance_factor = tolerance_factor
        self.losses = deque([], maxlen=reference_window + tolerance_window)
        self.window_size = reference_window + tolerance_window

    def _should_stop(self):
        writer = CallbackBase.log_writer_singleton
        assert writer is not None
        # track loss
        loss = writer.log_cache[f"loss/online/total/{self.to_short_interval_string()}"]
        assert isinstance(loss, float)
        self.losses.append(loss)

        # dont stop if training is just getting started
        if len(self.losses) < self.window_size:
            return False

        window = list(self.losses)
        reference_window = window[:self.reference_window]
        tolerance_window = window[self.reference_window:]
        assert len(tolerance_window) == self.tolerance_window

        reference_mean = np.mean(reference_window)
        tolerance_mean = np.mean(tolerance_window)
        reference_mean_with_tolerance = reference_mean * self.tolerance_factor
        if tolerance_mean >= reference_mean_with_tolerance:
            self.logger.info(f"loss diverged -> stop training")
            self.logger.info(f"reference_window={self.reference_window} reference_mean={reference_mean}")
            self.logger.info(f"tolerance_window={self.tolerance_window} tolerance_mean={tolerance_mean}")
            self.logger.info(
                f"tolerance_factor={self.tolerance_factor} "
                f"reference_mean_with_tolerance={reference_mean_with_tolerance}"
            )
            return True
        return False
