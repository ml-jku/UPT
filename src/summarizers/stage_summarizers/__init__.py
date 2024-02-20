from utils.factory import instantiate


def stage_summarizer_from_kwargs(kind, **kwargs):
    return instantiate(module_names=[f"summarizers.stage_summarizers.{kind}"], type_names=[kind], **kwargs)
