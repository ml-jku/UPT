from utils.factory import instantiate


def extractor_from_kwargs(kind, **kwargs):
    return instantiate(module_names=[f"models.extractors.{kind}"], type_names=[kind], **kwargs)
