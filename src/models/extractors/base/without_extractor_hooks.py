from contextlib import contextmanager

from models.extractors.base.forward_hook import StopForwardException


@contextmanager
def without_extractor_hooks(extractors):
    assert isinstance(extractors, (tuple, list))
    for pooling in extractors:
        pooling.disable_hooks()
    try:
        yield
    except StopForwardException:
        for pooling in extractors:
            pooling.enable_hooks()
        raise
    for pooling in extractors:
        pooling.enable_hooks()
