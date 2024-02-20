from .base.param_group_modifier import ParamGroupModifier


class VitNormLrScaleModifier(ParamGroupModifier):
    def __init__(self, scale, start_block_index=None):
        self.scale = scale
        self.start_block_index = start_block_index

    def get_properties(self, model, name, param):
        assert self.start_block_index is not None
        if self.start_block_index < 0:
            start_idx = self.start_block_index + len(model.blocks)
        else:
            start_idx = self.start_block_index

        block_indices = list(range(start_idx, len(model.blocks)))
        if name.startswith("block") and int(name.split(".")[1]) in block_indices and "norm" in name:
            return dict(lr_scale=self.scale)
        return {}

    def __str__(self):
        return f"{type(self).__name__}(scale={self.scale},start_block_index={self.start_block_index})"
