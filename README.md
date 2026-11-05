# local-llms-sourcing

A command-line tool for benchmarking local LLMs on your own hardware, and a crowdsourced dataset of the results.

Which model, quantization, and inference engine run best on *your* GPU? Published benchmarks rarely cover your exact setup. This project collects real-world results from many hardware configurations so it can give recommendations tailored to what you own.

## What it does

The CLI wraps the major model distributors and inference engines behind one interface. You choose what to test, it runs the benchmark, and you can optionally submit the results to a shared endpoint.

You can control:

- **Model**: pull from supported model distributors
- **Quantization**: pick the quantization level you want to evaluate
- **Speculative decoding**: turn it on or off
- **Inference engine**: choose which backend runs the model
- **Task**: run the model on different tasks

## Installation

Requires Python 3.10 or newer.

```sh
git clone https://github.com/buaiml/local-llms-sourcing.git
cd local-llms-sourcing
pip install -e .
```

This installs the `local-llms` command. To keep it isolated from your system Python, install into a virtual environment first (`python -m venv .venv && source .venv/bin/activate`), or use `pipx install -e .`.

To run the tests:

```sh
pip install pytest
pytest
```

## CLI usage

Only model and engine management are available so far; benchmarking and submission are still in progress.

### Download a model

```sh
local-llms download -m <org/name> [-q <quantization>]
```

Models come from Hugging Face. For repos that ship GGUF files, `-q/--quantization` is required and selects which files to fetch (case-insensitive, e.g. `Q4_K_M`). If you leave it out, or ask for one that doesn't exist, the error lists the quantizations available. For repos without GGUF files, the whole repo is downloaded.

```sh
local-llms download -m Qwen/Qwen2.5-7B-Instruct-GGUF -q Q4_K_M
```

Before downloading, the CLI prints the files and total size and checks there is enough free disk space. Interrupted downloads resume where they left off, and re-running the command for a model you already have is a no-op.

Gated or private repos need a Hugging Face token: set `HF_TOKEN` or log in with the Hugging Face CLI.

### List downloaded models

```sh
local-llms list --models
```

```
MODEL                          QUANT   SIZE    SOURCE
Qwen/Qwen2.5-7B-Instruct-GGUF  Q4_K_M  4.4 GB  huggingface
```

### Remove a model

```sh
local-llms remove -m <org/name> [-q <quantization>]
```

Without `-q`, every downloaded quantization of the model is removed.

### Install an inference engine

```sh
local-llms download -e llama.cpp [--version <version>] [--build <build>]
```

Supported engines: `llama.cpp`, installed from the prebuilt binaries in its [GitHub releases](https://github.com/ggml-org/llama.cpp/releases).

- `--version` picks a release tag such as `b12138` (or just `12138`). The default is the latest release.
- `--build` picks the hardware build, e.g. `cpu`, `cuda`, `rocm`, `vulkan`, or an exact build like `cuda-12.8`. By default it's chosen from your hardware: CUDA if `nvidia-smi` is present (the newest build your driver supports), ROCm if ROCm is installed, Metal on Apple Silicon, and CPU otherwise. An unknown build lists the ones available for your platform.

Downloads resume if interrupted and are verified against the release's SHA-256 checksums. Disk space is checked before downloading and again before extracting. Several versions and builds can be installed side by side, e.g. to compare CUDA against Vulkan.

After installing, the CLI runs `llama-server --version` and warns if the binary can't run on your system. The official Linux binaries need glibc 2.38 or newer (Ubuntu 24.04, Debian 13, or similar).

GitHub limits unauthenticated API requests to 60 per hour. If you hit the limit, set `GITHUB_TOKEN`.

### List installed engines

```sh
local-llms list --engines
```

### Update an engine

```sh
local-llms update -e llama.cpp
```

Installs the latest release for each build you have installed, then removes the older versions.

### Remove an engine

```sh
local-llms remove -e llama.cpp [--version <version>]
```

Without `--version`, every installed version and build is removed.

### Where files are stored

Models, engines, and a `manifest.json` index live in `$XDG_DATA_HOME/local-llms` (default `~/.local/share/local-llms`). Set `LOCAL_LLMS_HOME` to use a different directory.

## Crowdsourced results

After a run, you can submit your results to our endpoint. Submissions are aggregated across:

- GPUs and other accelerators
- Overall hardware setups (VRAM, system RAM, CPU, and so on)
- Inference engines
- Models, quantizations, and decoding settings

The aggregated data powers **personalized recommendations**: given your hardware, the project can suggest which model, quantization, and engine combination is likely to give you the best results.

Submitting is opt-in. Nothing is sent unless you choose to.

## Why

- Local LLM performance depends on many interacting choices, such as hardware, engine, quantization, and decoding strategy.
- Most benchmark data comes from a handful of popular setups.
- A large, community-sourced dataset can fill in the gaps and guide people toward configurations that work well on hardware like theirs.

## Status

Early development. Interfaces, supported engines, and the submission schema may change.
