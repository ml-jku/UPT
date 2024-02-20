import logging

LOWER_IS_BETTER_KEYS = [
    "loss",
    "delta",
]
HIGHER_IS_BETTER_KEYS = [
    "profiling/train_update_time",
    "correlation",
    "corerlation_time",
]
NEUTRAL_KEYS = [
    "optim",
    "profiling",
    "mask_ratio",
    "freezers",
    "transform_scale",
    "ctx",
    "loss_weight",
    "gradient",
    "detach",
    "confidence",
    "train_len",
    "test_len",
    "degree",
]


def is_neutral_key(metric_key):
    for higher_is_better_key in HIGHER_IS_BETTER_KEYS:
        if metric_key.startswith(higher_is_better_key):
            return False
    for lower_is_better_key in LOWER_IS_BETTER_KEYS:
        if metric_key.startswith(lower_is_better_key):
            return False
    for neutral_key in NEUTRAL_KEYS:
        if metric_key.startswith(neutral_key):
            return True
    return False


def higher_is_better_from_metric_key(metric_key):
    for higher_is_better_key in HIGHER_IS_BETTER_KEYS:
        if metric_key.startswith(higher_is_better_key):
            return True
    for lower_is_better_key in LOWER_IS_BETTER_KEYS:
        if metric_key.startswith(lower_is_better_key):
            return False
    logging.warning(f"{metric_key} has no defined behavior for higher_is_better -> using True")
    return True
