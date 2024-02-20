from pathlib import Path

import torch

from models.ssl_heads.masked_decoder import MaskedDecoder
from .base.initializer_base import InitializerBase


class PretrainedInitializer(InitializerBase):
    """ initialize with weights from an external, pretrained checkpoints (e.g. original facebook MAE checkpoints) """

    def __init__(self, weights_file, root_key=None, key_mapping=None, **kwargs):
        super().__init__(**kwargs)
        self.weights_file = weights_file
        self.weights_uri = Path(self.path_provider.model_path / weights_file).expanduser()
        assert self.weights_uri.exists() and self.weights_uri.is_file(), self.weights_uri.as_posix()
        self.key_mapping = key_mapping
        self.root_key = root_key

    def _get_model_kwargs(self):
        self.logger.info(f"loading ckpt kwargs for '{self.weights_uri}'")
        kwargs = dict(kind="vit.vit")
        # I-JEPA no CLS token
        if "ijepa" in self.weights_file:
            kwargs["cls_tokens"] = 0

        # ViT dimensions
        if "base16" in self.weights_file:
            return dict(patch_size=16, dim=768, num_attn_heads=12, depth=12, **kwargs)
        if "large16" in self.weights_file:
            return dict(patch_size=16, dim=1024, num_attn_heads=16, depth=24, **kwargs)
        if "huge16" in self.weights_file:
            return dict(patch_size=16, dim=1280, num_attn_heads=16, depth=32, **kwargs)
        if "huge14" in self.weights_file:
            return dict(patch_size=14, dim=1280, num_attn_heads=16, depth=32, **kwargs)

        sd = torch.load(self.weights_uri, map_location=torch.device("cpu"))
        if "ctor_kwargs" in sd:
            kwargs = sd["ctor_kwargs"]
        else:
            kwargs = {}
        self.logger.info(f"found kwargs: {kwargs}")
        return kwargs

    def init_weights(self, model):
        self.logger.info(f"loading weights from '{self.weights_uri}'")
        sd = torch.load(self.weights_uri, map_location=torch.device("cpu"))
        # unpack state_dict
        # - MLPlayground stores weights in "state_dict" field
        # - MAE stores weights in "model" field
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]
        # select model (e.g. used when student/teacher is stored in same checkpoint)
        if self.root_key is not None:
            sd = sd[self.root_key]

        #
        if isinstance(model, MaskedDecoder) and self.weights_file in [
            "mae_base16.pth", "mae_large16.pth", "mae_huge14.pth",  # MAE
            "mae_base16res448.pth", "mae_large16res448.pth",  # long sequence MAE
            "mae_base16res448e800.pth", "mae_large16res448e800.pth",  # long sequence MAE
        ]:
            for key in sd.keys():
                print(key)
            sd = {k: v for k, v in sd.items() if "decoder" in k}
        elif self.weights_file in [
            "mae_base16.pth", "mae_large16.pth", "mae_huge14.pth",  # MAE
            "mae_base16res448.pth", "mae_large16res448.pth",  # long sequence MAE
            "mae_base16res448e800.pth", "mae_large16res448e800.pth",  # long sequence MAE
        ]:
            sd = {k: v for k, v in sd.items() if "decoder" not in k and k != "mask_token"}
        elif "layergrafting" in self.weights_file:
            sd = {
                k.replace("module.momentum_encoder.", ""): v
                for k, v in sd.items()
                if k.startswith("module.momentum_encoder.") and "head" not in k
            }
        elif "mugs" in self.weights_file:
            sd = {k: v for k, v in sd.items() if not k.startswith("relation_blocks")}
        elif "ijepa" in self.weights_file:
            sd = {k.replace("module.", ""): v for k, v in sd.items()}

        # remap keys
        if self.key_mapping is not None:
            for old_key, new_key in self.key_mapping.items():
                sd[new_key] = sd.pop(old_key)

        model.load_state_dict(sd)
