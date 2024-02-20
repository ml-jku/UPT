from utils.factory import instantiate


def trainer_from_kwargs(kind, **kwargs):
    if "eval" in kind:
        return instantiate(module_names=[f"trainers.eval.{kind}"], type_names=[kind], **kwargs)
    return instantiate(module_names=[f"trainers.{kind}"], type_names=[kind.split(".")[-1]], **kwargs)
