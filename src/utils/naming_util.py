from kappadata.datasets import KDSubset


def join_names(name1, name2):
    if name1 is None:
        return name2
    assert name2 is not None
    return f"{name1}.{name2}"


def pascal_to_snake(pascal_case: str) -> str:
    """
    convert pascal/camel to snake case https://learn.microsoft.com/en-us/visualstudio/code-quality/ca1709?view=vs-2022
    "By convention, two-letter acronyms use all uppercase letters,
    and acronyms of three or more characters use Pascal casing."
    """
    if len(pascal_case) == 0:
        return ""
    snake_case = [pascal_case[0].lower()]
    upper_counter = 0
    for i in range(1, len(pascal_case)):
        if pascal_case[i].islower():
            snake_case += [pascal_case[i]]
            upper_counter = 0
        else:
            if upper_counter == 2:
                upper_counter = 0
            if upper_counter == 0:
                snake_case += ["_"]
            snake_case += [pascal_case[i].lower()]
            upper_counter += 1
    return "".join(snake_case)


def _type_name(obj, to_name_fn):
    if isinstance(obj, KDSubset):
        return _type_name(obj.dataset, to_name_fn)
    cls = type(obj)
    return to_name_fn(cls.__name__)


def lower_type_name(obj):
    return _type_name(obj, lambda name: name.lower())


def snake_type_name(obj):
    return _type_name(obj, pascal_to_snake)
