class StopForwardException(Exception):
    pass


class ForwardHook:
    def __init__(
            self,
            outputs: dict,
            output_name: str,
            raise_exception: bool = False,
    ):
        self.outputs = outputs
        self.output_name = output_name
        self.raise_exception = raise_exception
        self.enabled = True

    def __call__(self, _, __, output):
        if not self.enabled:
            return

        self.outputs[self.output_name] = output
        if self.raise_exception:
            raise StopForwardException()
