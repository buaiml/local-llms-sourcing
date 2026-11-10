import os

from local_llms_sourcing.env import load_dotenv


def test_load_dotenv(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# c\nA_TEST=1\nexport B_TEST='two words'\nC_TEST=\"x\"\nEXISTING_TEST=new\nbad line\n")
    monkeypatch.setenv("EXISTING_TEST", "old")
    for k in ("A_TEST", "B_TEST", "C_TEST"):
        monkeypatch.delenv(k, raising=False)
    assert load_dotenv(f) == f
    assert (os.environ["A_TEST"], os.environ["B_TEST"], os.environ["C_TEST"]) == ("1", "two words", "x")
    assert os.environ["EXISTING_TEST"] == "old"
    for k in ("A_TEST", "B_TEST", "C_TEST"):
        monkeypatch.delenv(k)


def test_load_dotenv_missing(tmp_path):
    assert load_dotenv(tmp_path / ".env") is None
