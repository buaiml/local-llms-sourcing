import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


def root_dir() -> Path:
    env = os.environ.get("LOCAL_LLMS_HOME")
    if env:
        return Path(env).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(data_home).expanduser() / "local-llms"


def models_dir() -> Path:
    return root_dir() / "models"


def manifest_path() -> Path:
    return root_dir() / "manifest.json"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def free_space(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


@dataclass
class ModelEntry:
    model: str
    quantization: str | None
    source: str
    revision: str
    size: int
    path: str
    files: list[str]


class Manifest:
    """JSON record of installed models, so list/remove don't scan the disk."""

    def __init__(self) -> None:
        self.path = manifest_path()
        self.models: list[ModelEntry] = []
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.models = [ModelEntry(**m) for m in data.get("models", [])]

    def find(self, model: str, quantization: str | None) -> ModelEntry | None:
        for entry in self.models:
            if entry.model == model and entry.quantization == quantization:
                return entry
        return None

    def add(self, entry: ModelEntry) -> None:
        self.models = [
            m
            for m in self.models
            if not (m.model == entry.model and m.quantization == entry.quantization)
        ]
        self.models.append(entry)
        self.save()

    def discard(self, entry: ModelEntry) -> None:
        self.models.remove(entry)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"models": [asdict(m) for m in self.models]}, indent=2))
        tmp.replace(self.path)
