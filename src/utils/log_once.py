import logging

_MESSAGE_KEYS = set()


def log_once(log_fn_or_message, key, level=logging.INFO):
    if key not in _MESSAGE_KEYS:
        if isinstance(log_fn_or_message, str):
            logging.log(level=level, msg=log_fn_or_message)
        else:
            log_fn_or_message()
        _MESSAGE_KEYS.add(key)
