from pathlib import Path

from .base.command_base import CommandBase


class CopyCommand(CommandBase):
    def __init__(self, src: str, dst: str, **kwargs):
        super().__init__(**kwargs)
        self.src = Path(self._resolve_string(src)).expanduser()
        self.dst = Path(self._resolve_string(dst)).expanduser()

    def __str__(self):
        return f"{type(self).__name__}(src={self.src}, dst={self.dst})"

    def execute(self):
        self.logger.info(f"copying '{self.src}' to '{self.dst}'")
        self.dst.parent.mkdir(exist_ok=True, parents=True)
        with open(self.src) as f:
            src = f.read()
        src = self._resolve_string(src)
        with open(self.dst, "w") as f:
            f.write(src)
        self.logger.info(f"copied '{self.src}' to '{self.dst}'")
