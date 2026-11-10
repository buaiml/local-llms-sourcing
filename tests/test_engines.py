import io
import shutil
import sys
import tarfile

import pytest

from local_llms_sourcing import engines
from local_llms_sourcing.cli import main
from local_llms_sourcing.engines import (
    MLX,
    VLLM,
    ArchivePlan,
    Asset,
    ExLlamaV3,
    Hardware,
    LlamaCpp,
    PipPlan,
    select_build,
)
from local_llms_sourcing.storage import Manifest

NAMES = [
    "cudart-llama-b100-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "cudart-llama-b100-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "llama-b100-bin-ubuntu-x64.tar.gz",
    "llama-b100-bin-ubuntu-arm64.tar.gz",
    "llama-b100-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "llama-b100-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "llama-b100-bin-ubuntu-rocm-10.0-x64.tar.gz",
    "llama-b100-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b100-bin-ubuntu-26.04-s390x.tar.gz",
    "llama-b100-bin-macos-arm64.tar.gz",
    "llama-b100-bin-win-cpu-x64.zip",
    "llama-b100-xcframework.zip",
]
ASSETS = [
    {"name": n, "browser_download_url": f"https://x/{n}", "size": 10, "digest": "sha256:ab"}
    for n in NAMES
]
LINUX = Hardware("ubuntu", "x64", "cpu")
LINUX_CUDA = Hardware("ubuntu", "x64", "cuda", (13, 4))
MAC = Hardware("macos", "arm64", "metal")


def test_builds_for_linux_x64():
    builds = LlamaCpp().builds(ASSETS, LINUX)
    assert set(builds) == {"cpu", "cuda-12.8", "cuda-13.4", "rocm-10.0", "vulkan"}
    assert [a.name for a in builds["cuda-13.4"]] == [
        "llama-b100-bin-ubuntu-cuda-13.4-x64.tar.gz",
        "cudart-llama-b100-bin-ubuntu-cuda-13.4-x64.tar.gz",
    ]
    assert builds["cpu"][0].sha256 == "ab"


def test_builds_other_platforms():
    assert set(LlamaCpp().builds(ASSETS, MAC)) == {"metal"}
    assert set(LlamaCpp().builds(ASSETS, Hardware("win", "x64", "cpu"))) == {"cpu"}
    assert set(LlamaCpp().builds(ASSETS, Hardware("ubuntu", "arm64", "cpu"))) == {"cpu"}


@pytest.mark.parametrize(
    "hw, requested, expected",
    [
        (LINUX, None, "cpu"),
        (LINUX_CUDA, None, "cuda-13.4"),
        (Hardware("ubuntu", "x64", "cuda", (12, 9)), None, "cuda-12.8"),
        (Hardware("ubuntu", "x64", "cuda", (11, 8)), None, "cpu"),
        (Hardware("ubuntu", "x64", "rocm"), None, "rocm-10.0"),
        (LINUX, "cuda", "cuda-13.4"),
        (LINUX, "cuda-12.8", "cuda-12.8"),
        (LINUX, "vulkan", "vulkan"),
    ],
)
def test_select_build(hw, requested, expected):
    builds = list(LlamaCpp().builds(ASSETS, LINUX))
    assert select_build("llama.cpp", builds, requested, hw) == expected


def test_select_unknown_build_lists_available():
    builds = list(LlamaCpp().builds(ASSETS, LINUX))
    with pytest.raises(engines.EngineError, match="Available: cpu, cuda-12.8"):
        select_build("llama.cpp", builds, "sycl", LINUX)


def test_gpu_only_engine_needs_explicit_build_without_gpu():
    with pytest.raises(engines.EngineError, match="pass --build"):
        select_build("vllm", VLLM().builds(LINUX), None, LINUX)
    assert select_build("vllm", VLLM().builds(LINUX), "cuda", LINUX) == "cuda"
    assert VLLM().builds(MAC) == []


@pytest.mark.parametrize(
    "hw, expected, requirement",
    [
        (MAC, "metal", "mlx-lm==1.0"),
        (LINUX, "cpu", "mlx-lm[cpu]==1.0"),
        (LINUX_CUDA, "cuda-13", "mlx-lm[cuda13]==1.0"),
        (Hardware("ubuntu", "x64", "cuda", (12, 8)), "cuda-12", "mlx-lm[cuda12]==1.0"),
    ],
)
def test_mlx_builds(hw, expected, requirement):
    mlx = MLX()
    build = select_build("mlx", mlx.builds(hw), None, hw)
    assert build == expected
    assert mlx.requirement(build, "1.0") == requirement


def test_pip_engine_resolve(monkeypatch):
    monkeypatch.setattr(
        engines, "_get_json", lambda url: {"info": {"version": "0.9", "requires_python": ">=3.0"}}
    )
    plan = VLLM().resolve(None, None, LINUX_CUDA)
    assert (plan.version, plan.build, plan.steps) == ("0.9", "cuda", [["vllm==0.9"]])
    assert plan.estimated


def test_pip_engine_rejects_unsupported_python(monkeypatch):
    monkeypatch.setattr(
        engines, "_get_json", lambda url: {"info": {"version": "0.9", "requires_python": "<3.0"}}
    )
    with pytest.raises(engines.EngineError, match="needs Python <3.0"):
        VLLM().resolve(None, None, LINUX_CUDA)


def test_exllamav3_picks_wheel_for_python_and_cuda(monkeypatch):
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    names = [
        f"exllamav3-1.0+cu128.torch2.8.0-{py}-{py}-linux_x86_64.whl",
        f"exllamav3-1.0+cu128.torch2.10.0-{py}-{py}-linux_x86_64.whl",
        f"exllamav3-1.0+cu132.torch2.13.0-{py}-{py}-linux_x86_64.whl",
        f"exllamav3-1.0+cu132.torch2.13.0-{py}-{py}-win_amd64.whl",
        "exllamav3-1.0+cu132.torch2.13.0-cp27-cp27-linux_x86_64.whl",
    ]
    rel = {
        "tag_name": "v1.0",
        "assets": [
            {"name": n, "browser_download_url": f"https://x/{n}", "size": 1, "digest": "sha256:cd"}
            for n in names
        ],
    }
    monkeypatch.setattr(engines, "_github", lambda path: rel)

    plan = ExLlamaV3().resolve(None, None, Hardware("ubuntu", "x64", "cuda", (12, 9)))
    assert (plan.version, plan.build) == ("1.0", "cuda-12.8")
    assert plan.steps == [
        ["torch==2.10.0", "--index-url", "https://download.pytorch.org/whl/cu128"],
        [f"https://x/{names[1]}#sha256=cd"],
    ]
    assert ExLlamaV3().resolve(None, None, LINUX_CUDA).build == "cuda-13.2"
    with pytest.raises(engines.EngineError, match="no prebuilt exllamav3"):
        ExLlamaV3().resolve(None, None, Hardware("ubuntu", "arm64", "cuda", (13, 4)))


def test_pip_plan_installs_into_venv(tmp_path, monkeypatch):
    calls = []

    def run(cmd, check=False):
        calls.append(cmd)
        return type("Proc", (), {"returncode": 0})()

    monkeypatch.setattr(engines.subprocess, "run", run)
    plan = PipPlan("1", "cuda", [], 0, steps=[["torch==2"], ["pkg==1"]])
    plan.install(tmp_path / "venv", lambda msg: None)
    assert calls[0][1:3] == ["-m", "venv"]
    assert calls[1][0] == str(engines._venv_python(tmp_path / "venv"))
    assert calls[1][-1] == "torch==2" and calls[2][-1] == "pkg==1"


def make_tarball(path, version):
    with tarfile.open(path, "w:gz") as tf:
        for name in ("llama-server", "libllama.so"):
            data = f"{version} {name}".encode()
            info = tarfile.TarInfo(f"llama-{version}/{name}")
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))


@pytest.fixture
def fake_release(tmp_path, monkeypatch):
    """Serve tarballs from disk instead of GitHub; set `release["version"]` to change latest."""
    monkeypatch.setenv("LOCAL_LLMS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(engines, "detect_hardware", lambda: LINUX)
    release = {"version": "b100"}

    def resolve(self, version, build, hw):
        version = version or release["version"]
        src = tmp_path / f"{version}.tar.gz"
        make_tarball(src, version)
        asset = Asset(f"llama-{version}-bin-ubuntu-x64.tar.gz", str(src), src.stat().st_size, None)
        return ArchivePlan(version, build or "cpu", [], asset.size, assets=[asset])

    monkeypatch.setattr(LlamaCpp, "resolve", resolve)
    monkeypatch.setattr(engines, "_fetch", lambda asset, dest: shutil.copy(asset.url, dest))
    monkeypatch.setattr(LlamaCpp, "check", lambda self, path: None)
    return release


def test_install_list_remove(fake_release, capsys):
    entry = engines.install("llama.cpp")
    server = engines.engines_dir() / "llama.cpp" / "b100-cpu" / "llama-server"
    assert server.read_text() == "b100 llama-server"
    assert entry.path == str(server.parent)
    assert not list((engines.engines_dir() / "llama.cpp" / ".downloads").iterdir())

    engines.install("llama.cpp")
    assert "already installed" in capsys.readouterr().out

    main(["list", "--engines"])
    assert "b100" in capsys.readouterr().out

    main(["remove", "-e", "llama.cpp", "--version", "100"])
    assert not server.parent.exists()
    assert Manifest().engines == []

    with pytest.raises(SystemExit):
        main(["remove", "-e", "llama.cpp"])


def test_failed_install_leaves_nothing_behind(fake_release, monkeypatch):
    def boom(self, target, out):
        target.mkdir(parents=True)
        raise engines.EngineError("pip install failed")

    monkeypatch.setattr(ArchivePlan, "install", boom)
    with pytest.raises(engines.EngineError):
        engines.install("llama.cpp")
    assert not (engines.engines_dir() / "llama.cpp" / "b100-cpu").exists()
    assert Manifest().engines == []


def test_update_replaces_old_version(fake_release):
    engines.install("llama.cpp", build="vulkan")
    fake_release["version"] = "b101"
    engines.update("llama.cpp")
    assert [(e.version, e.build) for e in Manifest().engines] == [("b101", "vulkan")]
    assert not (engines.engines_dir() / "llama.cpp" / "b100-vulkan").exists()


def test_cli_rejects_mismatched_flags():
    with pytest.raises(SystemExit):
        main(["download", "-m", "o/m", "--build", "cpu"])
    with pytest.raises(SystemExit):
        main(["download", "-e", "llama.cpp", "-q", "Q4_K_M"])
