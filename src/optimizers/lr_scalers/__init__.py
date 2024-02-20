from utils.factory import instantiate


def lr_scaler_from_kwargs(kind, **kwargs):
    return instantiate(
        module_names=[f"optimizers.lr_scalers.{kind}"],
        type_names=[kind],
        **kwargs
    )
