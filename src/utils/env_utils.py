import os


def env_flag_is_true(key):
    if key not in os.environ:
        return False
    return os.environ[key] in ["true", "True", "T"]
