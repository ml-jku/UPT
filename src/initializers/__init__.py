from utils.factory import instantiate


def initializer_from_kwargs(kind, **kwargs):
    return instantiate(
        module_names=[f"initializers.{kind}"],
        type_names=[kind.split(".")[-1]],
        **kwargs
    )
