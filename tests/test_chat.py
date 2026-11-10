import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from local_llms_sourcing import chat, engines
from local_llms_sourcing.chat import Stats, stream_chat
from local_llms_sourcing.storage import EngineEntry, Manifest, ModelEntry


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for t in ["Hel", "lo", " there"]:
            d = {"choices": [{"delta": {"content": t}}]}
            self.wfile.write(f"data: {json.dumps(d)}\n\n".encode())
        self.wfile.write(b'data: {"choices": [], "usage": {"completion_tokens": 3}}\n\n')
        self.wfile.write(b"data: [DONE]\n\n")


def test_stream_chat_collects_text_and_stats():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    seen = []
    reply, stats = stream_chat(
        f"http://127.0.0.1:{srv.server_port}/v1", "m", [{"role": "user", "content": "hi"}],
        lambda t, thinking: seen.append(t), lambda s: None,
    )
    srv.shutdown()
    assert reply == "Hello there" and seen == ["Hel", "lo", " there"]
    assert stats.tokens == 3 and stats.ttft is not None


def test_stats_line():
    assert Stats().line() == "TTFT waiting... | - tok/s | 0 tokens"
    s = Stats(ttft=0.5, tokens=11, elapsed=1.5)
    assert s.tps == 10.0 and s.line() == "TTFT 0.50s | 10.0 tok/s | 11 tokens"


def test_serve_commands(tmp_path):
    ll = engines.ENGINES["llama.cpp"].serve_command(tmp_path, "/m.gguf", 9)
    assert ll[1:3] == ["-m", "/m.gguf"] and ll[-1] == "9"
    sg = engines.ENGINES["sglang"].serve_command(tmp_path, "/m", 9)
    assert "sglang.launch_server" in sg and "--model-path" in sg
    with pytest.raises(engines.EngineError, match="no built-in server"):
        engines.ENGINES["exllamav3"].serve_command(tmp_path, "/m", 9)


def test_pickers_skip_exllamav3(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_LLMS_HOME", str(tmp_path))
    m = Manifest()
    m.add(EngineEntry("exllamav3", "1", "cuda-12.8", "x", 1, "/e"))
    m.add(EngineEntry("llama.cpp", "b1", "cpu", "x", 1, "/l"))
    m.add(ModelEntry("o/m", "Q4_K_M", "huggingface", "sha", 1, "/m", ["a-00001-of-00002.gguf", "a-00002-of-00002.gguf"]))
    assert chat.pick_engine(m, None, None).engine == "llama.cpp"
    assert chat.pick_model(m, "o/m", "q4_k_m").model == "o/m"
    assert chat.model_path(m.models[0]) == str(Path("/m") / "a-00001-of-00002.gguf")
