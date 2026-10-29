import re
import shutil
from pathlib import Path

from .storage import (
    Manifest,
    ModelEntry,
    dir_size,
    free_space,
    human_size,
    models_dir,
)

SPLIT_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


class ModelError(Exception):
    pass


def _matches_quant(filename: str, quant: str) -> bool:
    """True if `quant` (e.g. Q4_K_M) appears as a whole token run in the filename."""
    name = Path(filename).name
    stem = re.sub(r"\.gguf$", "", SPLIT_RE.sub(".gguf", name), flags=re.IGNORECASE)
    pattern = rf"(?<![A-Za-z0-9]){re.escape(quant)}(?![A-Za-z0-9])"
    return re.search(pattern, stem, re.IGNORECASE) is not None


def _available_quants(ggufs: list[str]) -> list[str]:
    found = set()
    for f in ggufs:
        m = re.search(
            r"(?<![A-Za-z0-9])((?:IQ|Q)\d[A-Za-z0-9_]*|BF16|F16|F32)(?![A-Za-z0-9])",
            Path(f).name,
            re.IGNORECASE,
        )
        if m:
            found.add(m.group(1).upper())
    return sorted(found)


def dir_name(model: str) -> str:
    return model.replace("/", "--")


def resolve(api, model: str, quantization: str | None):
    """Return (sha, [(filename, size)]) for the files to download."""
    try:
        info = api.model_info(model, files_metadata=True)
    except Exception as exc:
        raise ModelError(f"cannot fetch model '{model}' from Hugging Face: {exc}") from exc
    siblings = [(s.rfilename, s.size or 0) for s in info.siblings or []]
    ggufs = [(n, s) for n, s in siblings if n.lower().endswith(".gguf")]

    if quantization:
        chosen = [(n, s) for n, s in ggufs if _matches_quant(n, quantization)]
        if not chosen:
            avail = _available_quants([n for n, _ in ggufs])
            hint = f" Available: {', '.join(avail)}." if avail else " Repo has no GGUF files."
            raise ModelError(f"no files for quantization '{quantization}' in {model}.{hint}")
    elif ggufs:
        avail = _available_quants([n for n, _ in ggufs])
        raise ModelError(
            f"{model} has GGUF files; pass --quantization. Available: {', '.join(avail)}"
        )
    else:
        chosen = siblings
    if not chosen:
        raise ModelError(f"no files found in {model}")
    return info.sha, chosen


def download(model: str, quantization: str | None, out=print) -> ModelEntry:
    from huggingface_hub import HfApi, hf_hub_download

    manifest = Manifest()
    existing = manifest.find(model, quantization)
    if existing and all((Path(existing.path) / f).is_file() for f in existing.files):
        out(f"{model} ({quantization or 'full'}) already downloaded at {existing.path}")
        return existing

    api = HfApi()
    sha, files = resolve(api, model, quantization)
    total = sum(s for _, s in files)
    out(f"Downloading {len(files)} file(s) from {model}, {human_size(total)}:")
    for name, size in files:
        out(f"  {name}  {human_size(size)}")

    target = models_dir() / dir_name(model) / (quantization.upper() if quantization else "full")
    target.mkdir(parents=True, exist_ok=True)
    already = dir_size(target)
    needed = max(total - already, 0)
    free = free_space(target)
    if needed > free:
        raise ModelError(
            f"not enough disk space: need {human_size(needed)}, "
            f"only {human_size(free)} free at {target}"
        )

    for name, _ in files:
        # local_dir downloads resume from .cache/huggingface partial files and verify sizes/hashes
        hf_hub_download(model, name, revision=sha, local_dir=target)

    entry = ModelEntry(
        model=model,
        quantization=quantization.upper() if quantization else None,
        source="huggingface",
        revision=sha,
        size=sum((target / n).stat().st_size for n, _ in files),
        path=str(target),
        files=[n for n, _ in files],
    )
    manifest.add(entry)
    out(f"Done: {human_size(entry.size)} at {target}")
    return entry


def list_models() -> list[ModelEntry]:
    return sorted(Manifest().models, key=lambda m: (m.model, m.quantization or ""))


def remove(model: str, quantization: str | None = None) -> list[ModelEntry]:
    manifest = Manifest()
    targets = [
        m
        for m in manifest.models
        if m.model == model and (quantization is None or m.quantization == quantization.upper())
    ]
    if not targets:
        raise ModelError(f"model '{model}' not found; see `local-llms list --models`")
    for entry in targets:
        shutil.rmtree(entry.path, ignore_errors=True)
        manifest.discard(entry)
    parent = models_dir() / dir_name(model)
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
    return targets
