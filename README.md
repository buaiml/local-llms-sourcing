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

| Engine | `-e` | Installed from | Builds | Platforms |
| --- | --- | --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `llama.cpp` | prebuilt release binaries | `cpu`, `cuda-*`, `rocm-*`, `vulkan`, `metal`, ... | Linux, macOS, Windows |
| [vLLM](https://github.com/vllm-project/vllm) | `vllm` | PyPI | `cuda` | Linux |
| [SGLang](https://github.com/sgl-project/sglang) | `sglang` | PyPI | `cuda` | Linux |
| [MLX](https://github.com/ml-explore/mlx-lm) (`mlx-lm`) | `mlx` | PyPI | `metal`, `cpu`, `cuda-12`, `cuda-13` | Apple Silicon, Linux |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | `exllamav3` | prebuilt release wheels + PyTorch | `cuda-12.8`, `cuda-13.2` | Linux x64, Windows x64 |

- `--version` picks a release, e.g. `b12138` for llama.cpp or `0.33.0` for vLLM. The default is the latest release.
- `--build` picks the hardware build, either exactly (`cuda-12.8`) or by family (`cuda` picks the newest). By default it's chosen from your hardware: CUDA if `nvidia-smi` is present (the newest build your driver supports), ROCm if ROCm is installed, Metal on Apple Silicon, and CPU otherwise. GPU-only engines refuse to install without a detected GPU unless you pass `--build` explicitly. An unknown build lists the ones available.

Several versions and builds can be installed side by side, e.g. to compare CUDA against Vulkan.

**llama.cpp** downloads resume if interrupted and are verified against the release's SHA-256 checksums. Disk space is checked before downloading and again before extracting. The official Linux binaries need glibc 2.38 or newer (Ubuntu 24.04, Debian 13, or similar).

**Python engines** (vLLM, SGLang, MLX, ExLlamaV3) are each installed into their own virtual environment, created with the same Python that runs `local-llms`, so that Python must be one the engine supports (3.11+ for vLLM, SGLang, and MLX). pip shows its own progress and caches downloads, so re-running after a failure doesn't download everything again. Their full size isn't known until pip resolves dependencies, so the disk check uses an estimate (about 12-14 GB for vLLM and SGLang, which pull in PyTorch and CUDA libraries). ExLlamaV3 is installed from its prebuilt wheels, which bundle compiled CUDA kernels, together with the matching PyTorch CUDA build.

After installing, the CLI checks the engine actually starts (`llama-server --version`, or importing the Python package) and warns if it doesn't.

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
