# Task 03: Engine and model compatibility

## Goal

Make engine and model pairings reliable. Some models fail on some engines for reasons unrelated to hardware. A common case is chat templates: a Qwen model ships a Jinja template that llama.cpp's template handling can't parse or renders differently from what the model expects. These problems should be found and handled before a run starts, not discovered halfway through one.

## Problem

- Model and engine each assume things about the chat template, special tokens, stop tokens, and tool or reasoning syntax. When the assumptions differ, the result is an error, or worse, silently degraded output (wrong role markers, missing generation prompt, bad stop behavior).
- Silent degradation is the dangerous case for this project: it would produce wrong benchmark scores and tokens/sec numbers that get submitted as if they were valid.
- Working out compatibility by hand during a run doesn't scale across many models and engines.

## Scope

1. A **compatibility registry** recording known-good, known-bad, and known-fixable pairings.
2. A **preflight check** the CLI runs before any run: it validates the pairing and applies fixes automatically where it can.
3. **Template handling**: detect template problems and resolve them automatically.
4. A **test harness** that exercises model and engine pairings to build and refresh the registry.

## Compatibility registry

- Keyed by model (family, and specific repo or variant where needed), engine, and engine version range.
- Each entry records a status:
  - `ok`: works as is
  - `fixable`: works with a specific fix (see below)
  - `broken`: does not work, with the reason
  - `unknown`: not yet tested
- Fixable entries name the fix: a template override, an engine flag, or a different engine version.
- Stored as data files in the repo, versioned and reviewable, so contributions can extend it.
- The CLI can update the registry from our server without a new CLI release (design to be decided).

## Preflight check

Before a run, the CLI:

1. Looks up the model and engine pairing in the registry and applies any known fix.
2. For unknown pairings, runs the template checks below.
3. Reports the outcome: proceeding as is, proceeding with a named fix, or refusing with a clear explanation.

### Broken pairings still produce a result file

Some pairings can't be fixed. That is useful data, not a dead end. When a pairing is `broken`, or preflight finds a problem with no fix:

- The CLI explains the problem to the user, then stops the run, or lets the user try anyway with an explicit override.
- Either way, it still generates a result file with an outcome of `incompatible` (or `failed`, if a forced run errored), the model, engine, and hardware details, and the failure reason: the error, the template mismatch, and logs where relevant.
- The user can upload that file like any other. Failures are valuable because they tell us, and later users, which pairings don't work.
- The registry can be updated from submitted failure reports, so a pairing that breaks for many users is flagged `broken` for everyone.

## Template handling

- Extract the chat template from the model's metadata (e.g. GGUF metadata or `tokenizer_config.json`).
- Check whether the target engine can parse and render it.
- Render a fixed set of sample conversations through the engine and compare against a reference rendering (e.g. the Hugging Face `transformers` Jinja rendering). Cover: system prompt, multi-turn, generation prompt present, and any tool or reasoning syntax the model supports.
- On mismatch or parse failure, try fixes in order:
  1. A known template override from the registry
  2. A compatible template for the same model family
  3. Engine flags that change template handling
  4. If nothing works, report a clear message and record the failure in a result file (see above)
- Verify stop and special tokens match what the model expects.
- Record which template was actually used in the result file (see task 01), so results from overridden templates are distinguishable.

## Test harness

- A script that takes a list of models and engines, runs the preflight checks and a short smoke generation on each combination, and emits registry entries.
- Smoke generation checks that output is non-empty, terminates properly, and doesn't contain leaked template tokens.
- Runnable in CI on a schedule, and by contributors locally, so the registry can stay current as engines and models release.
- Failures are saved with logs for debugging.

## Interaction with other tasks

- Depends on task 02 for fetching models and engines.
- Adds fields to the result file from task 01: the template used, any fix applied, and the registry entry version consulted.
- Task 01's result file and endpoint must accept failed and incompatible runs, which have an outcome and failure reason but no performance metrics.
- Results from `fixable` pairings should be tagged so analysis can account for them.

## Acceptance criteria

- [ ] Registry format defined, with `ok` / `fixable` / `broken` / `unknown` statuses
- [ ] Preflight runs automatically before every run and reports its outcome
- [ ] A known failing case (e.g. a Qwen model's template on llama.cpp) is detected and resolved automatically via a registry fix
- [ ] Template rendering compared against a reference for a fixed set of sample conversations
- [ ] `broken` pairings don't run by default, but can be forced with an explicit override
- [ ] An unfixable or broken pairing still generates a result file with an outcome and failure reason, which the user can upload
- [ ] Test harness generates registry entries from model and engine combinations
- [ ] Result file records the template used and any fix applied

## Open questions

- Reference implementation for template rendering
- How the registry gets distributed and updated
- Which models and engines go into the first test matrix
- How to handle engines that apply the template server-side versus client-side
