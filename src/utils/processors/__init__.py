from utils.factory import instantiate


def processor_from_kwargs(kind, **kwargs):
    return instantiate(
        module_names=[f"utils.processors.{kind}"],
        type_names=[kind],
        **kwargs,
    )
