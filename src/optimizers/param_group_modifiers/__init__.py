from utils.factory import instantiate


def param_group_modifier_from_kwargs(kind, **kwargs):
    return instantiate(
        module_names=[f"optimizers.param_group_modifiers.{kind}"],
        type_names=[kind],
        **kwargs
    )
