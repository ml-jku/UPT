from .base.param_group_modifier import ParamGroupModifier


class ExcludeFromWdByNameModifier(ParamGroupModifier):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.param_was_found = False

    def get_properties(self, model, name, param):
        if name == self.name:
            self.param_was_found = True
            return dict(weight_decay=0.)
        return {}

    def __str__(self):
        return f"{type(self).__name__}(name={self.name})"

    def was_applied_successfully(self):
        return self.param_was_found
