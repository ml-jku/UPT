from pathlib import Path

import yaml

from .base.command_base import CommandBase


class CopyYamlCommand(CommandBase):
    def __init__(self, src: str, dst: str, prepend: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.src = Path(self._resolve_string(src)).expanduser()
        self.dst = Path(self._resolve_string(dst)).expanduser()
        self.preprend = prepend

    def __str__(self):
        return f"{type(self).__name__}(src={self.src}, dst={self.dst}, prepend={self.preprend})"

    def execute(self):
        self.logger.info(f"copying '{self.src}' to '{self.dst}' while prepending ({self.preprend})")
        self.dst.parent.mkdir(exist_ok=True, parents=True)
        with open(self.src) as f:
            src = f.read()
        src = self._resolve_string(src)
        src = yaml.safe_load(src)
        if self.preprend is not None:
            assert isinstance(src, dict)
            src = {**self.preprend, **src}
        with open(self.dst, "w") as f:
            yaml.safe_dump(src, f, sort_keys=False)
        self.logger.info(f"copied '{self.src}' to '{self.dst}'")
