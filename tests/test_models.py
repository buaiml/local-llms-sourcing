from types import SimpleNamespace

import pytest

from local_llms_sourcing import models
from local_llms_sourcing.cli import main
from local_llms_sourcing.storage import Manifest, ModelEntry


class FakeApi:
    def __init__(self, files):
        self.files = files

    def model_info(self, model, files_metadata=False):
        sibs = [SimpleNamespace(rfilename=n, size=s) for n, s in self.files]
        return SimpleNamespace(sha="abc123", siblings=sibs)


GGUFS = [
    ("m-Q4_K_M.gguf", 100),
    ("m-Q8_0.gguf", 200),
    ("m-Q4_K_S.gguf", 90),
    ("big-Q6_K-00001-of-00002.gguf", 5),
    ("big-Q6_K-00002-of-00002.gguf", 5),
    ("README.md", 1),
]


def test_resolve_exact_quant():
    _, files = models.resolve(FakeApi(GGUFS), "o/m", "q4_k_m")
    assert [n for n, _ in files] == ["m-Q4_K_M.gguf"]


def test_resolve_split_files():
    _, files = models.resolve(FakeApi(GGUFS), "o/m", "Q6_K")
    assert len(files) == 2


def test_resolve_unknown_quant_lists_available():
    with pytest.raises(models.ModelError, match="Available: .*Q4_K_M"):
        models.resolve(FakeApi(GGUFS), "o/m", "Q2_K")


def test_resolve_requires_quant_for_gguf_repo():
    with pytest.raises(models.ModelError, match="pass --quantization"):
        models.resolve(FakeApi(GGUFS), "o/m", None)


def test_resolve_non_gguf_repo_takes_all():
    _, files = models.resolve(FakeApi([("a.safetensors", 1), ("config.json", 1)]), "o/m", None)
    assert len(files) == 2


def test_list_and_remove(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_LLMS_HOME", str(tmp_path))
    d = tmp_path / "models" / "o--m" / "Q4_K_M"
    d.mkdir(parents=True)
    (d / "m.gguf").write_bytes(b"x" * 10)
    Manifest().add(ModelEntry("o/m", "Q4_K_M", "huggingface", "abc", 10, str(d), ["m.gguf"]))

    main(["list", "--models"])
    assert "o/m" in capsys.readouterr().out

    main(["remove", "-m", "o/m"])
    assert not d.exists()
    assert not (tmp_path / "models" / "o--m").exists()
    assert Manifest().models == []

    with pytest.raises(SystemExit):
        main(["remove", "-m", "o/m"])
