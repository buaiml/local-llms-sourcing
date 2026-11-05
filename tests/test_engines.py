import io
import shutil
import tarfile

import pytest

from local_llms_sourcing import engines
from local_llms_sourcing.cli import main
from local_llms_sourcing.engines import Asset, Hardware, LlamaCpp
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


def test_builds_for_linux_x64():
    builds = LlamaCpp().builds(ASSETS, LINUX)
    assert set(builds) == {"cpu", "cuda-12.8", "cuda-13.4", "rocm-10.0", "vulkan"}
    assert [a.name for a in builds["cuda-13.4"]] == [
        "llama-b100-bin-ubuntu-cuda-13.4-x64.tar.gz",
        "cudart-llama-b100-bin-ubuntu-cuda-13.4-x64.tar.gz",
    ]
    assert builds["cpu"][0].sha256 == "ab"


def test_builds_other_platforms():
    assert set(LlamaCpp().builds(ASSETS, Hardware("macos", "arm64", "metal"))) == {"metal"}
    assert set(LlamaCpp().builds(ASSETS, Hardware("win", "x64", "cpu"))) == {"cpu"}
    assert set(LlamaCpp().builds(ASSETS, Hardware("ubuntu", "arm64", "cpu"))) == {"cpu"}


@pytest.mark.parametrize(
    "hw, requested, expected",
    [
        (LINUX, None, "cpu"),
        (Hardware("ubuntu", "x64", "cuda", (13, 4)), None, "cuda-13.4"),
        (Hardware("ubuntu", "x64", "cuda", (12, 9)), None, "cuda-12.8"),
        (Hardware("ubuntu", "x64", "cuda", (11, 8)), None, "cpu"),
        (Hardware("ubuntu", "x64", "rocm"), None, "rocm-10.0"),
        (LINUX, "cuda", "cuda-13.4"),
        (LINUX, "cuda-12.8", "cuda-12.8"),
        (LINUX, "vulkan", "vulkan"),
    ],
)
def test_select_build(hw, requested, expected):
    llama = LlamaCpp()
    assert llama.select_build(llama.builds(ASSETS, LINUX), requested, hw) == expected


def test_select_unknown_build_lists_available():
    llama = LlamaCpp()
    with pytest.raises(engines.EngineError, match="Available: cpu, cuda-12.8"):
        llama.select_build(llama.builds(ASSETS, LINUX), "sycl", LINUX)


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
        name = f"llama-{version}-bin-ubuntu-x64.tar.gz"
        return version, build or "cpu", [Asset(name, str(src), src.stat().st_size, None)]

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
