"""Interactive chat against a locally started engine server, with live speed stats."""

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import engines
from .storage import EngineEntry, Manifest, ModelEntry


class ChatError(Exception):
    pass


# --- choosing what to chat with ---


def pick_model(manifest: Manifest, name: str | None, quant: str | None) -> ModelEntry:
    entries = manifest.models
    if name:
        entries = [m for m in entries if m.model == name]
    if quant:
        entries = [m for m in entries if m.quantization == quant.upper()]
    if not entries:
        raise ChatError("no matching downloaded model; see `local-llms list --models`")
    if len(entries) == 1:
        return entries[0]
    return _ask("Model", entries, lambda m: f"{m.model} ({m.quantization or 'full'})")


def pick_engine(manifest: Manifest, name: str | None, version: str | None) -> EngineEntry:
    entries = [e for e in manifest.engines if _can_serve(e.engine)]
    if name:
        backend = engines.get_engine(name)
        entries = [e for e in entries if e.engine == backend.name]
    if version:
        entries = [e for e in entries if e.version == version]
    if not entries:
        raise ChatError("no matching installed engine; see `local-llms list --engines`")
    if name and len({e.engine for e in entries}) == 1:  # several versions: newest
        return max(entries, key=lambda e: engines._version_tuple(e.version))
    if len(entries) == 1:
        return entries[0]
    return _ask("Engine", entries, lambda e: f"{e.engine} {e.version} ({e.build})")


def _can_serve(engine: str) -> bool:
    return getattr(engines.get_engine(engine), "server_module", True) is not None


def _ask(label: str, entries: list, show) -> object:
    if not sys.stdin.isatty():
        raise ChatError(f"several {label.lower()}s match; narrow it with flags")
    for i, e in enumerate(entries, 1):
        print(f"  {i}. {show(e)}")
    while True:
        answer = input(f"{label} [1-{len(entries)}]: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(entries):
            return entries[int(answer) - 1]


def model_path(entry: ModelEntry) -> str:
    ggufs = sorted(f for f in entry.files if f.lower().endswith(".gguf"))
    return str(Path(entry.path) / ggufs[0]) if ggufs else entry.path


# --- server lifecycle ---


class Server:
    def __init__(self, engine: EngineEntry, model: ModelEntry):
        backend = engines.get_engine(engine.engine)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.base = f"http://127.0.0.1:{self.port}/v1"
        self.cmd = backend.serve_command(Path(engine.path), model_path(model), self.port)
        self.log = tempfile.NamedTemporaryFile("w+", prefix="local-llms-server-", suffix=".log")
        self.proc: subprocess.Popen | None = None

    def start(self, timeout: float = 600) -> str:
        """Start the server, wait until it is ready, return the served model id."""
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise ChatError(f"server exited with code {self.proc.returncode}:\n{self.tail()}")
            try:
                with urllib.request.urlopen(f"{self.base}/models", timeout=2) as r:
                    data = json.load(r)
                if data.get("data"):
                    return data["data"][0]["id"]
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.5)
        raise ChatError(f"server not ready after {int(timeout)}s:\n{self.tail()}")

    def tail(self, n: int = 15) -> str:
        self.log.flush()
        self.log.seek(0)
        return "".join(self.log.readlines()[-n:]).rstrip() + f"\n(full log: {self.log.name})"

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --- streaming and metrics ---


@dataclass
class Stats:
    ttft: float | None = None  # seconds from sending the request to the first token
    tokens: int = 0
    elapsed: float = 0.0  # seconds since the request was sent

    @property
    def tps(self) -> float | None:
        """Tokens per second, counted from the first token."""
        if self.ttft is None or self.tokens < 2 or self.elapsed <= self.ttft:
            return None
        return (self.tokens - 1) / (self.elapsed - self.ttft)

    def line(self) -> str:
        ttft = f"{self.ttft:.2f}s" if self.ttft is not None else "waiting..."
        tps = f"{self.tps:.1f}" if self.tps is not None else "-"
        return f"TTFT {ttft} | {tps} tok/s | {self.tokens} tokens"


def stream_chat(base: str, model: str, messages: list[dict], on_text, on_stats) -> tuple[str, Stats]:
    """Send one chat request and stream it. Calls on_text(str) and on_stats(Stats) live."""
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions", body, {"Content-Type": "application/json"}
    )
    stats, reply = Stats(), []
    start = time.monotonic()
    last_ui = 0.0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                chunk = json.loads(line[5:])
                now = time.monotonic() - start
                stats.elapsed = now
                if usage := chunk.get("usage"):
                    stats.tokens = usage.get("completion_tokens", stats.tokens)
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    thinking = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if not (text or thinking):
                        continue
                    if stats.ttft is None:
                        stats.ttft = now
                    stats.tokens += 1  # one streamed chunk is about one token; usage corrects it
                    if text:
                        reply.append(text)
                    on_text(text or thinking, bool(thinking and not text))
                if now - last_ui > 0.1:
                    last_ui = now
                    on_stats(stats)
    except urllib.error.HTTPError as exc:
        raise ChatError(f"server error {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ChatError(f"lost connection to the server: {exc}") from exc
    stats.elapsed = time.monotonic() - start
    on_stats(stats)
    return "".join(reply), stats


# --- terminal UI ---


class StatusBar:
    """Pins a live status line to the bottom row using a terminal scroll region."""

    def __init__(self):
        self.on = sys.stdout.isatty()
        self.rows = shutil.get_terminal_size().lines

    def __enter__(self):
        if self.on:
            # make room, then confine scrolling to every row but the last
            sys.stdout.write(f"\n\033[A\0337\033[1;{self.rows - 1}r\0338")
            sys.stdout.flush()
        return self

    def __exit__(self, *exc):
        if self.on:
            sys.stdout.write(f"\0337\033[{self.rows};1H\033[2K\033[r\0338")
            sys.stdout.flush()

    def set(self, text: str) -> None:
        if self.on:
            sys.stdout.write(f"\0337\033[{self.rows};1H\033[2K\033[7m {text} \033[0m\0338")
            sys.stdout.flush()


def run(engine: EngineEntry, model: ModelEntry) -> None:
    server = Server(engine, model)
    print(f"Starting {engine.engine} {engine.version} with {model.model} ({model.quantization or 'full'})...")
    try:
        served = server.start()
        print("Ready. Type /exit to quit, /clear to reset the conversation.\n")
        history: list[dict] = []
        with StatusBar() as bar:
            bar.set(f"{engine.engine} | {model.model} | send a message")
            while True:
                try:
                    user = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                if not user:
                    continue
                if user in ("/exit", "/quit"):
                    return
                if user == "/clear":
                    history.clear()
                    print("Conversation cleared.")
                    continue
                history.append({"role": "user", "content": user})
                shown = {"thinking": False}

                def on_text(t: str, thinking: bool) -> None:
                    if thinking != shown["thinking"]:
                        shown["thinking"] = thinking
                        sys.stdout.write("\n[thinking] " if thinking else "\n")
                    sys.stdout.write(t)
                    sys.stdout.flush()

                try:
                    reply, stats = stream_chat(server.base, served, history, on_text, lambda s: bar.set(s.line()))
                except KeyboardInterrupt:
                    print("\n(interrupted)")
                    history.pop()
                    continue
                except ChatError as exc:
                    history.pop()
                    print(f"\nerror: {exc}", file=sys.stderr)
                    continue
                print()
                history.append({"role": "assistant", "content": reply})
                if not bar.on:
                    print(stats.line())
    finally:
        server.stop()
