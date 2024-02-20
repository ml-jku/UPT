from utils.factory import instantiate


def callback_from_kwargs(kind, **kwargs):
    return instantiate(
        module_names=[
            f"callbacks.{kind}",
            f"callbacks.checkpoint_callbacks.{kind}",
            f"callbacks.default_callbacks.{kind}",
            f"callbacks.monitor_callbacks.{kind}",
            f"callbacks.offline_callbacks.{kind}",
            f"callbacks.online_callbacks.{kind}",
            f"callbacks.retroactive_callbacks.{kind}",
            f"callbacks.visualization.{kind}",
        ],
        type_names=[kind],
        **kwargs,
    )
