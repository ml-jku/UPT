import logging


class CommandBase:
    def __init__(self, stage_id, variables=None):
        self.logger = logging.getLogger(type(self).__name__)
        self.stage_id = stage_id
        self.variables = variables or {}
        self.variables["stage_id"] = self.stage_id

    def __repr__(self):
        return str(self)

    def __str__(self):
        raise NotImplementedError

    def _resolve_string(self, string):
        for key, value in self.variables.items():
            string = string.replace("{{" + key + "}}", value)
        return string

    def execute(self):
        raise NotImplementedError
