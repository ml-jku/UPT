from utils.factory import instantiate


def finalizer_from_kwargs(kind, **kwargs):
    return instantiate(module_names=[f"models.extractors.finalizers.{kind}"], type_names=[kind], **kwargs)
