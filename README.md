# cspace — transformational-creativity research harness

Harness for the pilot described in *Transformational Creativity for Autonomous
Research*: agents question commitments in a small causal-LM architecture
"box," implement reformulations, and test them with compute-capped smoke tests.

**Status:** Gate 1 (measurement and reliable execution) is built. The research
loop (OpenAI-backed workers, fixed scheduler) runs end to end, and a first draft of
the box is in `boxes/`. Next: the calibration cohort (short and long runs for every
candidate) and mechanism probes.

## The box (conceptual space)

`boxes/lm_state_compute_v0.json` is the draft K/C dependency graph for the pilot.
`boxes/lm_state_compute_v0.md` is a readable dossier rendered from it.

- 60 nodes: 4 definitions; 9 active commitments of the starting decoder plus the
  baseline method; 5 consequences and theorems; 23 empirical claims, each with the
  scale of its evidence; 18 known alternatives as background.
- Every source cites an arXiv ID checked against its abstract page. Research
  provenance: `boxes/research/` (deep-research report and citation sweep, Nov 2026).
- Statuses: `established`, `supported` (author-reported, abstract-verified),
  `conditional`, `disputed`, `unresolved`. Almost all evidence is at 70M-8B; the
  editable prior "results transfer to 5-30M" is part of the context, not a fact.

```
python -m cspace box render boxes/lm_state_compute_v0.json              # dossier (markdown)
python -m cspace box render boxes/lm_state_compute_v0.json --format dag # exactly what DAG agents see
python -m cspace box render boxes/lm_state_compute_v0.json --format flat
python -m cspace box query  boxes/lm_state_compute_v0.json c.kv_memory  # dependents / lost support
```

## Research loop

`cspace/agents/`: one campaign = one condition (`dag` or `flat`) with its own archive.

- **Wave** (four job slots, chosen from an evidence snapshot frozen at wave start):
  continuation of the oldest queued follow-up (or a fallback challenge of the
  incumbent); a challenge from the incumbent; a challenge from the least recently
  expanded valid state; and a reconstruction from a clean harness (observations only,
  logged retrievals).
- **Job**: information tools, then `submit_proposal`, which records the commitment,
  change, mechanism, prediction, control, test and scope before results. Then checks
  and a smoke test against the baseline control, with up to 2 repairs and an optional
  ablation run. The job ends with `record_reflection`: observations, interpretation,
  evidence class, next question, and an optional precommitted follow-up. Each origin
  allows 2 continuations.
- The **DAG** arm gets typed nodes, `query_box`, and validated `space_patch`
  operations. The **flat** arm gets the same content as a prose register with free-text
  revisions. Prompts are otherwise identical.
- Every LLM call is charged to the inference ledger (input, cached and output tokens).
  Transcripts are stored in the archive.

```
python -m cspace llm-ping --model gpt-5.4-mini           # uses OPENAI_API_KEY from the environment
python -m cspace campaign init pilot-dag --condition dag --model gpt-6-astra
python -m cspace campaign run pilot-dag --waves 4        # resumable
python -m cspace campaign status pilot-dag
python -m cspace campaign job pilot-dag j001_1 --transcript
```

## Layout

| Path | Role |
| --- | --- |
| `cspace/data/` | Corpus download, BPE tokenizer, deterministic token streams |
| `cspace/seeds/baseline_decoder/` | Starting candidate: Llama-style decoder (RoPE, RMSNorm, SwiGLU, tied embeddings) |
| `cspace/evaluator/` | Locked evaluator: lint, engineering checks, compute-capped training, probes, pacing, isolated runner |
| `cspace/agents/` | Research loop: LLM client, worker tools, wave scheduler, formulation patches |
| `cspace/calibration.py` | Stage A calibration cohort |
| `cspace/seeds/controls/` | Known control architectures for calibration |
| `boxes/` | Box v0 (60 nodes, 69 verified sources), rendered dossier, literature-search records |
| `cspace/archive/` | Content-addressed artifact store, SQLite index, cost ledgers |
| `cspace/box.py`, `cspace/schemas/space.schema.json` | K/C box records: validation, acyclicity, suspension queries |
| `cspace/schemas/` | JSON schemas for candidates, experiments, spaces |
| `data/`, `archive/` | Generated state (git-ignored) |

## Candidate contract

A candidate is a bundle directory with `model.py` (required), `train_config.json`
and `NOTES.md` (optional).

```python
def build_model(vocab_size: int, context_length: int) -> torch.nn.Module: ...
# forward(idx: LongTensor[B, T]) -> logits [B, T, vocab_size]
#   or {"logits": ..., "aux_loss": scalar}; any T <= context_length
```

- Imports are limited to `torch` and a few standard-library modules. No file or network
  access, reseeding, or monkeypatching (`cspace/evaluator/lint.py`). This is a guardrail,
  not a security sandbox.
- `train_config.json` may set `lr`, `weight_decay`, `beta1`, `beta2`, `warmup_frac` and
  `min_lr_frac` within fixed bounds. Batch size, context length, data and schedule shape
  are fixed by the task.
- Engineering checks (`cspace/evaluator/checks.py`) must pass: future-token perturbation
  and gradient tests, prefix consistency (no dependence on how many tokens follow), batch
  independence, eval-mode statelessness, finite gradients, tiny-batch overfit, and
  deterministic initialization.

## Budgets

Budgets are in **training FLOPs**, measured by executing the model under
`torch.utils.flop_counter`, so recurrent or adaptive computation is charged as executed.
The learning-rate schedule always spans the full horizon. A smoke test stops at an
earlier checkpoint fraction, so short and long runs share the same schedule prefix.
Attention is counted under the math SDPA kernel on every device, so a budget means the
same thing on CPU and GPU (baseline: 731.2 GFLOP per step of 32 × 512 tokens).

## Mechanism probes

After a completed run the evaluator records `probes` (evaluation only, no training):
`copy` shows a random token sequence twice and reports loss on each half, `copy_gain` and
second-half argmax accuracy at delays 8/32/128/256 (capped at half the context);
`position_loss` reports dev loss by context-position bucket. They are diagnostics shown to
agents next to the baseline's values, not definitions of usefulness.

## Hardware safety

`--throttle 0.5` (on `evaluate`, `replay`, `campaign init`, `cohort init`) sleeps after
each step so the GPU is busy half the time. `--temp-limit 80` reads `nvidia-smi` every 50
steps and pauses training until the GPU is 5 °C below the limit; a run that stays hot for
30 minutes ends with status `overheat`. Pacing changes wall-clock time only: results are
bitwise identical with or without it. `--device cpu` avoids the GPU entirely (slow).
Tests never use the GPU unless `CSPACE_GPU_TESTS=1`; on Windows the device is hidden with
`CUDA_VISIBLE_DEVICES=-1` (an empty value does not hide it).

## Calibration cohort (Stage A)

```
python -m cspace cohort init calib1 --source pilot-dag --source pilot-flat --cap 10 --throttle 0.5
python -m cspace cohort run calib1            # enroll -> smoke -> screen -> long -> report (resumable)
python -m cspace cohort status calib1
```

A cohort has its own archive under `archive/cohorts/<name>/` and a spec frozen by hash at
creation. Enrollment is outcome-blind: agent candidates in creation order, engineering-valid
only (`checks.passed` is the only result read), about half of the slots reserved for
reformulations (the agent's pre-result `change_type`, or researcher overrides). Known
controls in `cspace/seeds/controls/` (recurrent depth, depth-weighted averaging, wide-shallow)
and the baseline are added and reported separately. Every member gets the smoke test
(identical prior smoke runs in the source campaign are reused) and a long run with fresh
seeds, whatever its screening decision. Screening decisions (advance / hold / reject) are
written to `screening.json` and frozen before any long run is allowed. The report gives
paired deltas, Spearman correlations overall and by stratum, the missed-useful and
useful-among-rejects fractions with explicit numerators and denominators, learning-curve
crossings, and a smoke-vs-long scatter plot when matplotlib is installed.

## Commands

```
python -m cspace prepare-data --corpus tinystories       # or fineweb-edu
python -m cspace profile                                  # throughput at several sizes
python -m cspace seed-archive                             # archive the baseline
python -m cspace check cspace/seeds/baseline_decoder      # lint + engineering checks
python -m cspace evaluate --candidate <id> --horizon-flops 2.3e14 --checkpoints 0.25,0.5,1.0
python -m cspace replay <experiment-id> --tolerance 0     # must reproduce exactly
python -m cspace show | ledger
python -m cspace freeze-evaluator                         # lock evaluator source before campaigns
python -m pytest -q
```

Evaluations run in a subprocess with a hard timeout and an environment stripped of
API keys and tokens. The confirm split is refused unless `--allow-sealed` is passed.

## Measured on this machine (RTX 4070 Laptop, 8 GB, bf16, no torch.compile)

| Baseline size | Params (total / non-emb) | Train tokens/s | Minutes to 20 tokens/param |
| --- | --- | --- | --- |
| d256 L6 | 5.9M / 4.8M | ~187k | ~10 |
| d384 L8 | 15.7M / 14.2M | ~83k | ~63 |
| d512 L8 | 27.8M / 25.7M | ~57k | ~164 |

Vocabulary 4,096, context 512, batch 32 × 512 tokens. TinyStories debug task: 46.1M
train / 2.2M dev / 2.2M confirm tokens.

**Determinism.** On CUDA the default memory-efficient attention kernel has a
nondeterministic backward pass. Replays differed by up to 0.08 nats at 76 steps. The
evaluator therefore pins cuDNN attention, which was bitwise reproducible in testing at
about 95% of default speed, and falls back to the math kernel (about 2× slower) for
candidates cuDNN cannot run. FLOPs are always counted under the math SDPA kernel, on
every device, so neither the fallback nor the device changes budgets (fused kernels have
device-specific or missing FLOP formulas; the CPU flash kernel was not counted at all). The kernel used is recorded as `attention_backend`.

## Known limitations

- Lint and process isolation do not stop a determined adversarial candidate. Audits still
  read the code.
- The FLOP counter ignores elementwise operations, the usual convention. Candidates that
  move work into custom elementwise loops are undercharged. Audits should check this.
- Within a training window, documents are packed with an end-of-text separator and
  attention may cross document boundaries. This applies identically to every candidate.
