from utils.factory import instantiate


def command_from_kwargs(kind, **kwargs):
    return instantiate(
        module_names=[f"utils.commands.{kind}"],
        type_names=[kind.split(".")[-1]],
        **kwargs
    )
