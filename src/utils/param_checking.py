import collections.abc
from itertools import repeat
from pathlib import Path


# adapted from timm (timm/models/layers/helpers.py)
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            assert len(x) == n
            return x
        return tuple(repeat(x, n))

    return parse


def _is_ntuple(n):
    def check(x):
        return isinstance(x, tuple) and len(param) == n

    return check

def to_ntuple(x, n):
    return _ntuple(n=n)(x)

def is_ntuple(x, n):
    return _is_ntuple(n=n)(x)


to_2tuple = _ntuple(2)
is_2tuple = _is_ntuple(2)


def float_to_integer_exact(f):
    assert f.is_integer()
    return int(f)


def check_exclusive(*args):
    return sum(arg is not None for arg in args) == 1


def check_inclusive(*args):
    return sum(arg is not None for arg in args) in [0, len(args)]


def check_at_least_one(*args):
    return sum(arg is not None for arg in args) > 0


def check_at_most_one(*args):
    return sum(arg is not None for arg in args) <= 1


def to_path(path):
    if path is not None and not isinstance(path, Path):
        return Path(path).expanduser()
    return path


def to_list_of_values(list_or_item, default_value=None):
    if list_or_item is None:
        if default_value is None:
            return []
        else:
            return [default_value]
    if not isinstance(list_or_item, (tuple, list)):
        return [list_or_item]
    return list_or_item
