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
