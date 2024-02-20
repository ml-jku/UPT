from utils.factory import instantiate


def summary_summarizer_from_kwargs(kind, **kwargs):
    return instantiate(module_names=[f"summarizers.summary_summarizers.{kind}"], type_names=[kind], **kwargs)
