from utils.factory import instantiate


def freezer_from_kwargs(kind, **kwargs):
    return instantiate(
        module_names=[f"freezers.{kind}"],
        type_names=[kind.split(".")[-1]],
        **kwargs
    )
