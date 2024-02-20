from utils.factory import instantiate


def early_stopper_from_kwargs(kind, **kwargs):
    return instantiate(module_names=[f"trainers.early_stoppers.{kind}"], type_names=[kind], **kwargs)
