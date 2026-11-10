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
from dataclasses import dataclass, field
from pathlib import Path

from packaging.specifiers import SpecifierSet

from .storage import EngineEntry, Manifest, dir_size, engines_dir, free_space, human_size

GB = 1024**3


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


def _get_json(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        hint = " (rate limited; set GITHUB_TOKEN)" if exc.code in (403, 429) else ""
        raise EngineError(f"HTTP error {exc.code} for {url}{hint}") from exc
    except urllib.error.URLError as exc:
        raise EngineError(f"cannot reach {url}: {exc.reason}") from exc


def _github(path: str):
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return _get_json(f"https://api.github.com/{path}", headers)


def select_build(engine: str, builds: list[str], requested: str | None, hw: Hardware) -> str:
    """Pick a build by name or family (e.g. "cuda" -> newest cuda-*), or from detected hardware."""
    if not builds:
        raise EngineError(f"no {engine} builds for {hw.os}-{hw.arch}")

    def family(prefix: str) -> list[str]:
        names = [b for b in builds if b == prefix or b.startswith(prefix + "-")]
        return sorted(names, key=_version_tuple)

    if requested:
        if requested in builds:
            return requested
        if matches := family(requested):
            return matches[-1]
        raise EngineError(
            f"no '{requested}' build of {engine} for {hw.os}-{hw.arch}. "
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
        f"no build of {engine} matches detected hardware ({hw.accel}); "
        f"pass --build to install anyway. Available: {', '.join(sorted(builds))}"
    )


@dataclass
class Plan:
    """What an engine install will fetch, and how to put it in place."""

    version: str
    build: str
    items: list[tuple[str, int | None]]  # (description, size) shown before installing
    size: int  # bytes needed on disk
    estimated: bool = False

    def install(self, target: Path, out) -> None:
        raise NotImplementedError


# --- prebuilt release archives (llama.cpp) ---


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


@dataclass
class ArchivePlan(Plan):
    assets: list[Asset] = field(default_factory=list)

    def install(self, target: Path, out) -> None:
        downloads = target.parent / ".downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        archives = []
        for a in self.assets:
            dest = downloads / a.name
            if not dest.exists():
                _fetch(a, dest)
            archives.append(dest)

        extracted = sum(_extracted_size(p) for p in archives)
        free = free_space(target.parent)
        if extracted > free:
            raise EngineError(
                f"not enough disk space: need {human_size(extracted)} to extract, "
                f"only {human_size(free)} free at {target.parent}"
            )
        for p in archives:
            _extract(p, target)
        for p in archives:
            p.unlink()


class LlamaCpp:
    """llama.cpp prebuilt binaries from GitHub releases (tags like b12138)."""

    name = "llama.cpp"
    source = "github:ggml-org/llama.cpp"
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
            asset = _asset(a)
            if m.group(1):
                runtimes[variant] = asset
            else:
                main[variant] = [asset]
        # CUDA builds ship the CUDA runtime libraries as a separate archive
        for variant, rt in runtimes.items():
            if variant in main:
                main[variant].append(rt)
        return main

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

    def resolve(self, version: str | None, build: str | None, hw: Hardware) -> Plan:
        rel = self.release(version)
        builds = self.builds(rel["assets"], hw)
        chosen = select_build(self.name, list(builds), build, hw)
        assets = builds[chosen]
        return ArchivePlan(
            version=rel["tag_name"],
            build=chosen,
            items=[(a.name, a.size) for a in assets],
            size=sum(a.size for a in assets),
            assets=assets,
        )

    def check(self, path: Path) -> str | None:
        """Run the installed server binary; return an error message if it can't start."""
        exe = path / ("llama-server.exe" if os.name == "nt" else "llama-server")
        if not exe.is_file():
            return f"{exe.name} not found in the release archive"
        return _run_check([str(exe), "--version"])


def _asset(a: dict) -> Asset:
    digest = a.get("digest") or ""
    sha256 = digest.removeprefix("sha256:") if digest.startswith("sha256:") else None
    return Asset(a["name"], a["browser_download_url"], a["size"], sha256)


def _run_check(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout).strip().splitlines()
        return lines[-1] if lines else f"exit code {proc.returncode}"
    return None


# --- Python packages installed into a per-engine venv (vLLM, SGLang, MLX, ExLlamaV3) ---


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _check_python(engine: str, requires_python: str | None) -> None:
    current = platform.python_version()
    if requires_python and not SpecifierSet(requires_python).contains(current, prereleases=True):
        raise EngineError(
            f"{engine} needs Python {requires_python}, but local-llms is running on {current}"
        )


@dataclass
class PipPlan(Plan):
    # each step is one `pip install` invocation, run in order
    steps: list[list[str]] = field(default_factory=list)

    def install(self, target: Path, out) -> None:
        out(f"Creating virtual environment at {target}")
        subprocess.run([sys.executable, "-m", "venv", str(target)], check=True)
        python = str(_venv_python(target))
        for args in self.steps:
            cmd = [python, "-m", "pip", "install", "--disable-pip-version-check", *args]
            # pip's own output (progress, errors) goes straight to the terminal
            sys.stdout.flush()
            if subprocess.run(cmd).returncode != 0:
                raise EngineError(f"pip install failed for {' '.join(args)}; see the output above")


class PipEngine:
    """An engine distributed as a PyPI package; builds map to pip requirements."""

    name: str
    package: str
    module: str  # imported by the post-install check
    # rough installed size per build family (torch, CUDA libraries, etc.), for the disk check
    sizes: dict[str, int]

    @property
    def source(self) -> str:
        return f"pypi:{self.package}"

    def builds(self, hw: Hardware) -> list[str]:
        raise NotImplementedError

    def normalize_version(self, version: str) -> str:
        return version.removeprefix("v")

    def resolve(self, version: str | None, build: str | None, hw: Hardware) -> Plan:
        url = f"https://pypi.org/pypi/{self.package}/{version + '/' if version else ''}json"
        info = _get_json(url)
        if not info:
            raise EngineError(f"no {self.name} release '{version}' on PyPI")
        version = info["info"]["version"]
        _check_python(self.name, info["info"].get("requires_python"))
        chosen = select_build(self.name, self.builds(hw), build, hw)
        size = self.sizes.get(chosen) or self.sizes[chosen.split("-")[0]]
        return PipPlan(
            version=version,
            build=chosen,
            items=[(f"pip install {self.package}=={version} ({chosen}) and dependencies", None)],
            size=size,
            estimated=True,
            steps=[[self.requirement(chosen, version)]],
        )

    def requirement(self, build: str, version: str) -> str:
        return f"{self.package}=={version}"

    def check(self, path: Path) -> str | None:
        return _run_check([str(_venv_python(path)), "-c", f"import {self.module}"])


class VLLM(PipEngine):
    name = "vllm"
    package = "vllm"
    module = "vllm"
    sizes = {"cuda": 12 * GB}

    def builds(self, hw):
        # PyPI wheels are Linux-only and built against CUDA
        return ["cuda"] if hw.os == "ubuntu" else []


class SGLang(PipEngine):
    name = "sglang"
    package = "sglang"
    module = "sglang"
    sizes = {"cuda": 14 * GB}

    def builds(self, hw):
        return ["cuda"] if hw.os == "ubuntu" else []


class MLX(PipEngine):
    name = "mlx"
    package = "mlx-lm"
    module = "mlx_lm"
    sizes = {"metal": 1 * GB, "cpu": 1 * GB, "cuda": 5 * GB}
    EXTRAS = {"metal": None, "cpu": "cpu", "cuda-12": "cuda12", "cuda-13": "cuda13"}

    def builds(self, hw):
        if hw.os == "macos" and hw.arch == "arm64":
            return ["metal"]
        if hw.os == "ubuntu":
            return ["cpu", "cuda-12", "cuda-13"]
        return []

    def requirement(self, build, version):
        extra = self.EXTRAS[build]
        return f"{self.package}[{extra}]=={version}" if extra else f"{self.package}=={version}"


class ExLlamaV3(PipEngine):
    """ExLlamaV3 from prebuilt GitHub wheels; the PyPI package needs the CUDA toolkit to compile."""

    name = "exllamav3"
    package = "exllamav3"
    module = "exllamav3"
    repo = "turboderp-org/exllamav3"
    sizes = {"cuda": 8 * GB}
    WHEEL_RE = re.compile(
        r"^exllamav3-[\d.]+\+cu(\d+)\.torch([\d.]+)-(cp\d+)-cp\d+-(linux_x86_64|win_amd64)\.whl$"
    )

    @property
    def source(self) -> str:
        return f"github:{self.repo}"

    def wheels(self, assets: list[dict], hw: Hardware) -> dict[str, tuple[str, Asset]]:
        """Map build (cuda-12.8, ...) to (torch version, wheel) for this platform and Python."""
        plat = {("ubuntu", "x64"): "linux_x86_64", ("win", "x64"): "win_amd64"}.get(
            (hw.os, hw.arch)
        )
        py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        best: dict[str, tuple[str, Asset]] = {}
        for a in assets:
            m = self.WHEEL_RE.match(a["name"])
            if not m or m.group(3) != py_tag or m.group(4) != plat:
                continue
            cu = m.group(1)
            build = f"cuda-{cu[:-1]}.{cu[-1]}"
            torch = m.group(2)
            # several torch versions per CUDA version; take the newest
            if build not in best or _version_tuple(torch) > _version_tuple(best[build][0]):
                best[build] = (torch, _asset(a))
        return best

    def resolve(self, version, build, hw):
        if version:
            rel = _github(f"repos/{self.repo}/releases/tags/v{self.normalize_version(version)}")
            if not rel:
                raise EngineError(f"no {self.name} release 'v{self.normalize_version(version)}'")
        else:
            rel = _github(f"repos/{self.repo}/releases/latest")
            if not rel:
                raise EngineError(f"cannot find a {self.name} release")
        wheels = self.wheels(rel["assets"], hw)
        if not wheels:
            raise EngineError(
                f"no prebuilt {self.name} {rel['tag_name']} wheel for {hw.os}-{hw.arch} "
                f"and Python {platform.python_version()}"
            )
        chosen = select_build(self.name, list(wheels), build, hw)
        torch, wheel = wheels[chosen]
        cu = "cu" + "".join(str(p) for p in _version_tuple(chosen))
        wheel_req = f"{wheel.url}#sha256={wheel.sha256}" if wheel.sha256 else wheel.url
        return PipPlan(
            version=self.normalize_version(rel["tag_name"]),
            build=chosen,
            items=[
                (f"torch=={torch} from the PyTorch {cu} index", None),
                (wheel.name, wheel.size),
                ("remaining dependencies from PyPI", None),
            ],
            size=self.sizes["cuda"],
            estimated=True,
            steps=[
                [f"torch=={torch}", "--index-url", f"https://download.pytorch.org/whl/{cu}"],
                [wheel_req],
            ],
        )


ENGINES = {e.name: e for e in [LlamaCpp(), VLLM(), SGLang(), MLX(), ExLlamaV3()]}


def get_engine(name: str):
    backend = ENGINES.get(name) or ENGINES.get(name.lower())
    if not backend:
        raise EngineError(f"unknown engine '{name}'. Supported: {', '.join(ENGINES)}")
    return backend


def install(name: str, version: str | None = None, build: str | None = None, out=print) -> EngineEntry:
    backend = get_engine(name)
    hw = detect_hardware()
    plan = backend.resolve(version, build, hw)

    manifest = Manifest()
    existing = manifest.find_engine(backend.name, plan.version, plan.build)
    if existing and Path(existing.path).is_dir():
        out(f"{backend.name} {plan.version} ({plan.build}) already installed at {existing.path}")
        return existing

    approx = "~" if plan.estimated else ""
    out(
        f"Installing {backend.name} {plan.version} ({plan.build} build, detected {hw.accel}), "
        f"{approx}{human_size(plan.size)}:"
    )
    for desc, size in plan.items:
        out(f"  {desc}  {human_size(size)}" if size is not None else f"  {desc}")
    out(f"Destination: {engines_dir() / backend.name / f'{plan.version}-{plan.build}'}")

    root = engines_dir() / backend.name
    root.mkdir(parents=True, exist_ok=True)
    partial = sum(f.stat().st_size for f in (root / ".downloads").glob("*.part"))
    needed = max(plan.size - partial, 0)
    free = free_space(root)
    if needed > free:
        raise EngineError(
            f"not enough disk space: need {approx}{human_size(needed)}, "
            f"only {human_size(free)} free at {root}"
        )

    target = root / f"{plan.version}-{plan.build}"
    shutil.rmtree(target, ignore_errors=True)
    try:
        plan.install(target, out)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise

    entry = EngineEntry(
        engine=backend.name,
        version=plan.version,
        build=plan.build,
        source=backend.source,
        size=dir_size(target),
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
