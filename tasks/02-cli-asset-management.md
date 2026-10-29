# Task 02: CLI and asset management

## Goal

Build the first draft of the CLI, focused on getting and managing everything a test run needs: model weights, inference engines, and benchmarks. A user should be able to specify a model, quantization, and engine, and have the CLI fetch whatever is missing, check there is room for it, and let them list and delete assets later.

Actually running tests and producing result files is a separate task. This task covers acquiring, storing, and managing the pieces.

## Scope

1. Model weights: search, download, list, delete.
2. Inference engines: install, list, update, remove.
3. Benchmarks: download, list, delete.
4. Storage management: check free space before downloading, show usage, clean up.

## Model weights

- Specify a model by name, quantization, and any other variant details.
- Wrap multiple model sources behind one interface. Hugging Face is required; add a few others (e.g. Ollama registry, direct URL) as the design allows. Each source is a pluggable backend.
- Resolve the request to concrete files (e.g. the right GGUF for a quantization) and show the user what will be downloaded and its size before starting.
- Support resumable downloads, and verify checksums where the source provides them.
- Skip the download if the exact weights are already present.
- Commands to list downloaded models (with size on disk) and delete them.

## Inference engines

- Wrap the engines we want to compare (e.g. llama.cpp, vLLM, others to be decided). Each engine is a pluggable backend with its own install logic.
- When a run names an engine, use the installed copy if there is one. Otherwise fetch and install it automatically, after confirming with the user.
- Engines live in CLI-managed directories, not system-wide, so removal is clean.
- Record engine name and version, since the result file needs them (see task 01).
- Commands to list installed engines and versions, install a specific version, and remove an engine.
- Handle hardware-specific builds (CUDA, ROCm, Metal, CPU-only), choosing based on detected hardware.

## Benchmarks

- Download benchmark definitions and datasets from their sources.
- Commands to list available and downloaded benchmarks, download one, and delete one.
- Track benchmark name and version, since the result file records both.
- Where a benchmark can be run in parts, expose the parts so they can be fetched and selected individually.

## Storage management

- All assets live under one CLI-managed root directory, configurable by the user, with subdirectories for models, engines, and benchmarks.
- Before any download, compare the required size against free space on the target disk and refuse, with a clear message, if it won't fit. Account for temporary space (partial downloads, extraction).
- A command that shows total usage, broken down by models, engines, and benchmarks.
- A single place to list everything downloaded and remove any item.
- A manifest tracks what is installed (name, version, source, size, path), so listing and deleting don't depend on scanning the disk.

## Example usage (draft)

```bash
# models
cli model download <model> --quant Q4_K_M --source huggingface
cli model list
cli model delete <model> --quant Q4_K_M

# engines
cli engine install llama.cpp --version <version>
cli engine list
cli engine remove llama.cpp

# benchmarks
cli benchmark download <benchmark>
cli benchmark list
cli benchmark delete <benchmark>

# storage
cli storage status
```

Command names are a draft. Rename freely.

## Acceptance criteria

- [ ] Specify a model and quantization, and the CLI downloads the matching weights from Hugging Face
- [ ] At least one additional model source works behind the same interface
- [ ] Specifying an engine uses the installed copy, or installs it if missing
- [ ] Benchmarks can be downloaded, listed, and deleted
- [ ] Models and engines can be listed and deleted
- [ ] Downloads check free disk space first and fail clearly when there isn't enough
- [ ] Interrupted downloads resume
- [ ] `storage status` shows usage by category
- [ ] A manifest accurately reflects what is installed

## Open questions

- Which engines to support first
- Which model sources to support beyond Hugging Face
- Which benchmarks to support first, and where they come from
- Implementation language for the CLI
- Default root directory per OS
