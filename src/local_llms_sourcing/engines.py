import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .storage import EngineEntry, Manifest, engines_dir, free_space, human_size


class EngineError(Exception):
    pass


@dataclass
class Asset:
    name: str
    url: str
    size: int
    sha256: str | None


@dataclass
class Hardware:
    os: str  # ubuntu, macos, win (llama.cpp release naming)
    arch: str  # x64, arm64
    accel: str  # cuda, rocm, metal, cpu
    cuda_version: tuple[int, ...] | None = None


def _version_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", s))


def detect_hardware() -> Hardware:
    system = platform.system()
    os_name = {"Linux": "ubuntu", "Darwin": "macos", "Windows": "win"}.get(system)
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if not os_name or not arch:
        raise EngineError(f"unsupported platform: {system} {machine}")

    if os_name == "macos":
        return Hardware(os_name, arch, "metal" if arch == "arm64" else "cpu")
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            ).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        m = re.search(r"CUDA Version:\s*([\d.]+)", out)
        if m:
            return Hardware(os_name, arch, "cuda", _version_tuple(m.group(1)))
    if shutil.which("rocminfo") or Path("/opt/rocm").is_dir():
        return Hardware(os_name, arch, "rocm")
    return Hardware(os_name, arch, "cpu")


def _github(path: str):
    req = urllib.request.Request(
        f"https://api.github.com/{path}", headers={"Accept": "application/vnd.github+json"}
    )
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        hint = " (rate limited; set GITHUB_TOKEN)" if exc.code in (403, 429) else ""
        raise EngineError(f"GitHub API error {exc.code} for {path}{hint}") from exc
    except urllib.error.URLError as exc:
        raise EngineError(f"cannot reach GitHub: {exc.reason}") from exc


class LlamaCpp:
    """llama.cpp prebuilt binaries from GitHub releases (tags like b12138)."""

    name = "llama.cpp"
    repo = "ggml-org/llama.cpp"
    ASSET_RE = re.compile(
        r"^(cudart-)?llama-(?:b\d+-)?bin-(ubuntu|macos|win)(?:-(.+?))?-(x64|arm64)\.(?:tar\.gz|zip)$"
    )

    def builds(self, assets: list[dict], hw: Hardware) -> dict[str, list[Asset]]:
        """Map build name (cpu, cuda-13.4, vulkan, ...) to the assets it needs for this OS/arch."""
        main: dict[str, list[Asset]] = {}
        runtimes: dict[str, Asset] = {}
        for a in assets:
            m = self.ASSET_RE.match(a["name"])
            if not m or m.group(2) != hw.os or m.group(4) != hw.arch:
                continue
            variant = m.group(3)
            if variant is None:
                variant = "metal" if hw.os == "macos" and hw.arch == "arm64" else "cpu"
            digest = a.get("digest") or ""
            asset = Asset(
                a["name"],
                a["browser_download_url"],
                a["size"],
                digest.removeprefix("sha256:") if digest.startswith("sha256:") else None,
            )
            if m.group(1):
                runtimes[variant] = asset
            else:
                main[variant] = [asset]
        # CUDA builds ship the CUDA runtime libraries as a separate archive
        for variant, rt in runtimes.items():
            if variant in main:
                main[variant].append(rt)
        return main

    def select_build(self, builds: dict[str, list[Asset]], requested: str | None, hw: Hardware) -> str:
        if not builds:
            raise EngineError(f"no {self.name} builds for {hw.os}-{hw.arch} in this release")

        def family(prefix: str) -> list[str]:
            names = [b for b in builds if b == prefix or b.startswith(prefix + "-")]
            return sorted(names, key=_version_tuple)

        if requested:
            if requested in builds:
                return requested
            if matches := family(requested):
                return matches[-1]
            raise EngineError(
                f"no '{requested}' build of {self.name} for {hw.os}-{hw.arch}. "
                f"Available: {', '.join(sorted(builds))}"
            )

        candidates = family(hw.accel)
        if hw.accel == "cuda" and hw.cuda_version:
            # the driver only runs binaries built against its CUDA version or older
            candidates = [b for b in candidates if _version_tuple(b)[:2] <= hw.cuda_version[:2]]
        if candidates:
            return candidates[-1]
        if "cpu" in builds:
            return "cpu"
        raise EngineError(
            f"no build of {self.name} matches detected hardware ({hw.accel}); "
            f"pass --build. Available: {', '.join(sorted(builds))}"
        )

    def normalize_version(self, version: str) -> str:
        return version if version.startswith("b") else f"b{version}"

    def release(self, version: str | None) -> dict:
        if version:
            tag = self.normalize_version(version)
            rel = _github(f"repos/{self.repo}/releases/tags/{tag}")
            if not rel:
                raise EngineError(f"no {self.name} release '{tag}'")
            return rel
        for rel in _github(f"repos/{self.repo}/releases?per_page=20") or []:
            if re.fullmatch(r"b\d+", rel["tag_name"]) and rel["assets"]:
                return rel
        raise EngineError(f"cannot find a recent {self.name} release with binaries")

    def resolve(self, version: str | None, build: str | None, hw: Hardware):
        """Return (version, build, [Asset]) to install."""
        rel = self.release(version)
        builds = self.builds(rel["assets"], hw)
        chosen = self.select_build(builds, build, hw)
        return rel["tag_name"], chosen, builds[chosen]

    def check(self, path: Path) -> str | None:
        """Run the installed server binary; return an error message if it can't start."""
        exe = path / ("llama-server.exe" if os.name == "nt" else "llama-server")
        if not exe.is_file():
            return f"{exe.name} not found in the release archive"
        try:
            proc = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            return str(exc)
        if proc.returncode != 0:
            lines = (proc.stderr or proc.stdout).strip().splitlines()
            return lines[-1] if lines else f"exit code {proc.returncode}"
        return None


ENGINES = {e.name: e for e in [LlamaCpp()]}


def get_engine(name: str):
    backend = ENGINES.get(name) or ENGINES.get(name.lower())
    if not backend:
        raise EngineError(f"unknown engine '{name}'. Supported: {', '.join(ENGINES)}")
    return backend


def _fetch(asset: Asset, dest: Path) -> None:
    """Download with resume from a .part file, then verify the sha256 digest."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > asset.size:
        part.unlink()
        have = 0
    if have < asset.size:
        req = urllib.request.Request(asset.url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                mode = "ab" if have and resp.status == 206 else "wb"
                done = have if mode == "ab" else 0
                with part.open(mode) as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if sys.stderr.isatty():
                            pct = done * 100 // max(asset.size, 1)
                            print(f"\r  {asset.name}  {pct}%", end="", file=sys.stderr)
            if sys.stderr.isatty():
                print(file=sys.stderr)
        except (urllib.error.URLError, OSError) as exc:
            raise EngineError(f"download of {asset.name} failed: {exc}; re-run to resume") from exc

    if asset.sha256:
        h = hashlib.sha256()
        with part.open("rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        if h.hexdigest() != asset.sha256:
            part.unlink()
            raise EngineError(f"checksum mismatch for {asset.name}; deleted, re-run to retry")
    part.replace(dest)


def _extracted_size(archive: Path) -> int:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            return sum(i.file_size for i in zf.infolist())
    with tarfile.open(archive) as tf:
        return sum(m.size for m in tf.getmembers() if m.isfile())


def _extract(archive: Path, target: Path) -> None:
    """Extract into target, flattening a single top-level directory (llama-b1234/...)."""
    staging = target.with_name(target.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
    else:
        with tarfile.open(archive) as tf:
            if hasattr(tarfile, "data_filter"):
                tf.extractall(staging, filter="data")
            else:
                tf.extractall(staging)
    entries = list(staging.iterdir())
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
    target.mkdir(parents=True, exist_ok=True)
    for item in root.iterdir():
        shutil.move(str(item), str(target / item.name))
    shutil.rmtree(staging)


def install(name: str, version: str | None = None, build: str | None = None, out=print) -> EngineEntry:
    backend = get_engine(name)
    hw = detect_hardware()
    version, build, assets = backend.resolve(version, build, hw)

    manifest = Manifest()
    existing = manifest.find_engine(backend.name, version, build)
    if existing and Path(existing.path).is_dir():
        out(f"{backend.name} {version} ({build}) already installed at {existing.path}")
        return existing

    total = sum(a.size for a in assets)
    out(f"Installing {backend.name} {version} ({build} build, detected {hw.accel}), {human_size(total)}:")
    for a in assets:
        out(f"  {a.name}  {human_size(a.size)}")

    root = engines_dir() / backend.name
    downloads = root / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    partial = sum(
        (downloads / (a.name + ".part")).stat().st_size
        for a in assets
        if (downloads / (a.name + ".part")).exists()
    )
    needed = max(total - partial, 0)
    free = free_space(downloads)
    if needed > free:
        raise EngineError(
            f"not enough disk space: need {human_size(needed)} to download, "
            f"only {human_size(free)} free at {downloads}"
        )

    archives = []
    for a in assets:
        dest = downloads / a.name
        if not dest.exists():
            _fetch(a, dest)
        archives.append(dest)

    extracted = sum(_extracted_size(p) for p in archives)
    free = free_space(root)
    if extracted > free:
        raise EngineError(
            f"not enough disk space: need {human_size(extracted)} to extract, "
            f"only {human_size(free)} free at {root}"
        )
    target = root / f"{version}-{build}"
    shutil.rmtree(target, ignore_errors=True)
    for p in archives:
        _extract(p, target)
    for p in archives:
        p.unlink()

    entry = EngineEntry(
        engine=backend.name,
        version=version,
        build=build,
        source=f"github:{backend.repo}",
        size=sum(f.stat().st_size for f in target.rglob("*") if f.is_file() and not f.is_symlink()),
        path=str(target),
    )
    manifest.add(entry)
    out(f"Done: {human_size(entry.size)} at {target}")
    if problem := backend.check(target):
        out(f"warning: {backend.name} is installed but does not run on this system: {problem}")
    return entry


def list_engines() -> list[EngineEntry]:
    return sorted(Manifest().engines, key=lambda e: (e.engine, _version_tuple(e.version), e.build))


def remove(name: str, version: str | None = None) -> list[EngineEntry]:
    backend = get_engine(name)
    if version:
        version = backend.normalize_version(version)
    manifest = Manifest()
    targets = [
        e
        for e in manifest.engines
        if e.engine == backend.name and (version is None or e.version == version)
    ]
    if not targets:
        raise EngineError(f"engine '{name}' not installed; see `local-llms list --engines`")
    for entry in targets:
        shutil.rmtree(entry.path, ignore_errors=True)
        manifest.discard(entry)
    if not any(e.engine == backend.name for e in manifest.engines):
        shutil.rmtree(engines_dir() / backend.name, ignore_errors=True)
    return targets


def update(name: str, out=print) -> list[EngineEntry]:
    """Install the latest release for each installed build, then remove older versions."""
    backend = get_engine(name)
    installed = [e for e in Manifest().engines if e.engine == backend.name]
    if not installed:
        raise EngineError(f"{backend.name} is not installed; use `local-llms download -e {name}`")
    updated = []
    for build in sorted({e.build for e in installed}):
        entry = install(backend.name, None, build, out=out)
        manifest = Manifest()
        for old in [e for e in manifest.engines if e.engine == backend.name and e.build == build]:
            if old.version != entry.version:
                shutil.rmtree(old.path, ignore_errors=True)
                manifest.discard(old)
                out(f"Removed {backend.name} {old.version} ({build})")
        updated.append(entry)
    return updated
