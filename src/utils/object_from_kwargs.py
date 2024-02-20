def objects_from_kwargs(kwargs):
    if kwargs is None:
        return {}

    result = {}
    for k, v in kwargs.items():
        if isinstance(v, dict):
            # if no factory type is supplied -> derive from key
            # e.g. mask generators are usually unique and defined via the "mask_generator" key
            # which can be easily converted to the factory
            factory_type = v.pop("factory_type", key_to_factory_type(k))
            if factory_type is not None and "kind" in v:
                result[k] = object_from_kwargs(factory_type=factory_type, **v)
            else:
                result[k] = objects_from_kwargs(v)
        else:
            result[k] = v
    return result


def key_to_factory_type(key):
    if "mask_generator" in key:
        return "mask_generator"
    return None


def object_from_kwargs(factory_type, **kwargs):
    raise NotImplementedError
