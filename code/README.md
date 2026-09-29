# MP1 code — installation and usage

## Final frozen submission

The final checkpoint bundle is runs/final-submission/. It contains the depth-10 hybrid predictor and scored 1.52468 BPB on the full test split with CPU FP32. Its selected validation BPB is 1.506880. The checkpoint SHA256 is 7382455e5313495cd186f6773ad4eba6e313acf89773495650cd6999079129d1; the final test run took 40.108 seconds. Depth-10 preflight measured a maximum working set of 1,801,285,632 bytes (1.68 GiB), below the 4 GiB limit. The bundle is 57.52 MiB including the checkpoint and model code, below the 64 MiB limit. Run from the code directory: python evaluate.py --checkpoint runs/final-submission/checkpoint.pt --split test --device cpu --precision fp32 --threads 3.

## 2026-09-29 paired depth continuation

An authorized, bounded experiment compares the current 8-layer network with a
10-layer expansion. `prepare_depth.py` appends two residual blocks, setting only
their attention output and MLP down projections to zero. Initial predictions
match the original model exactly; the additional branches can subsequently learn.
Width 192, six heads, dropout 0.1, training tables and all mixture settings stay
fixed. Both branches start from `runs/hybrid-adaptive-selected/checkpoint.pt`.

The fixed plan is in `runs/depth_audit/plan.json`: at most 1000 updates per
branch (2000 total), batch 32, seed 31, CPU FP32 with three threads, AdamW
weight decay 0.1, LR 5e-5 to 5e-6 with 50 warmup steps, EMA 0.99. Validation
uses the complete hybrid predictor with the unchanged scorer, separately for
ordinary and EMA weights. A paired stop at the same update count applies if
either branch has two validation events without improvement. Existing best
weights always remain eligible.

`train.py --train-component neural` selects the original neural cross-entropy
objective for this experiment. The default `--train-component model` preserves
existing training behavior. Repeat the same component flag on resume. Optimizers
are rebuilt for both new stages, then restored with RNG/EMA state at subsequent
boundaries. Embedded ancestry preserves the previous training-time cost even
when the inference bundle does not have a metrics file.

The first two updates are the real smoke run and count toward the 1000-update
limit. Training then resumes at shared boundaries 400, 800 and 1000, with
restart snapshots every 20 updates. A chained SHA256 of sampled window starts
checks equal training samples across architectures and resume boundaries.
Equal targets do not imply equal compute: the deeper network costs more per step.

Eighteen engineering checks passed before the run. All prior checkpoints,
data/tokenizer and evaluator remain protected by hashes in `runs/depth_audit/`.
Actual commands, exit codes and wall times are recorded in `commands.json`;
training logs are named train_depth{depth}_step{boundary}.log. The paired run selected the depth-10 EMA checkpoint at step 1000: validation BPB 1.506879794. The frozen test-scored copy is runs/final-submission/, with test BPB 1.52468. The adaptive checkpoint below is the starting point for this depth experiment, not the final submission.

```powershell
.\.venv\Scripts\python.exe prepare_depth.py
.\.venv\Scripts\python.exe run_depth_experiment.py
.\.venv\Scripts\python.exe finalize_depth_report.py
```

These bounded orchestration scripts refuse existing output locations. The
full plan, outcome and report are under `runs/depth_audit/`; no test scoring or
additional mixture search is part of this experiment. Codex substantially
assisted with implementation, validation, execution and documentation.

## 2026-09-29 adaptive mixture experiment

This bounded follow-up adds three optional settings to `hybrid.py`: a
prefix-dependent n-gram weight (`ngram_gate`), recency-weighted phrase copying
(`copy_half_life`), and reduced copying when observed continuations disagree
(`copy_agreement`). All default to zero, preserving previous checkpoint behavior.
They use only current-window causal inputs and fixed training statistics.
The n-gram gate uses relative maximum predicted probabilities as a heuristic;
these confidence scores are not assumed to be calibrated correctness rates.

Exactly 12 configurations were declared before full-validation selection.
The selected configuration is `ngram_gate=0.5`, `copy_half_life=0`,
`copy_agreement=0.5`: moderate adaptive statistics weighting and continuation
agreement help, while recency weighting was not selected. The unchanged
evaluator reproduced **1.508208363922** identically three times, compared with
**1.509641562687** for the previous mixture: a reduction of **0.001433199 BPB**
(**0.09494%**). This is a small improvement and does not reach 1.3.
Original-evaluator resource verification and acceptance are recorded in
[`runs/adaptive_audit/summary.json`](runs/adaptive_audit/summary.json) and
[`REPORT.md`](runs/adaptive_audit/REPORT.md).

At CPU / FP32 / 3 threads, median scoring time is **56.722 s** versus
baseline **15.618 s** (**3.632x**), peak working set **1.337 GiB**, and the full
uncompressed inference bundle **54.134 MiB**. All measured limits pass on the
full validation set. At this adaptive-mixture stage, the selected bundle was `runs/hybrid-adaptive-selected/`; the later paired depth experiment superseded it for final submission. The previous
`runs/hybrid-selected/` bundle and all historical checkpoints are retained.
All 112 neural/statistical state tensors are bit-identical to the previous
model. New neural updates, new training targets and statistics refits are all
zero. The 12 configurations share component computation during selection;
they are not 12 neural training runs. Selection scoring took **56.441 s**,
the selection process including smoke checks took **70.722 s**, and separate
original-scorer/resource verification took **392.431 s**. Development and
engineering checks are additional; failed preliminary attempts are disclosed
in the audit report. Existing training and search costs remain recorded in the
earlier experiment reports.

Sixteen correctness checks passed, including adaptive causality, independent
windows, normalization, short-prefix equivalence and conflicting phrase
continuations. A real training-prefix smoke check compared shared search
calculations against actual model outputs for all 12 settings. No test-set
scoring, external training data or pretrained weights were used. Data,
tokenizer, `common.py`, `evaluate.py`, `student.py` and `train.py` are unchanged
from the start of this follow-up.

```powershell
.\.venv\Scripts\python.exe fit_adaptive.py --source runs/hybrid-selected/checkpoint.pt --run-dir runs/hybrid-adaptive-v1 --threads 3
.\.venv\Scripts\python.exe finalize_adaptive.py
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/hybrid-adaptive-selected/checkpoint.pt --split validation --device cpu --precision fp32 --threads 3
```

The first two commands refuse existing outputs. For another search use a new
`--run-dir`; `finalize_adaptive.py` also accepts `--run-dir`, `--audit-dir` and
`--bundle-dir`. Its audit directory must contain the protected-file hash
manifest and correctness evidence; for direct score reproduction use the
last command. Backups and all attempt logs are under `runs/adaptive_audit/`.
Codex substantially assisted with these changes, checks, experiment and report.

## 2026-09-28 hybrid predictor experiment

The new predictor is implemented in `hybrid.py` and keeps the best student
network's weights unchanged. It combines the neural probabilities with compact
statistics fitted only on the supplied training text, a continuous cache of
already observed tokens within each input window, and optional exact phrase
continuation within that same window. All input-dependent temporary state is
recomputed per call; no evaluation-window information is retained between calls.
The only persistent derived cache is a bigram table made from fixed training
statistics.

The bounded development search uses 20 initial mixtures, 18 longer-phrase /
temperature combinations, then 6 phrase-copy strengths. Final full
validation BPB improves from **1.602857188** to **1.509641563**; this does **not**
reach the aspirational 1.3 target. Independent original-evaluator scores,
resource comparisons, and controls are recorded in
[`runs/hybrid_audit/REPORT.md`](runs/hybrid_audit/REPORT.md) and
[`summary.json`](runs/hybrid_audit/summary.json).

The unchanged evaluator reproduced the final BPB identically three times.
Median CPU scoring time was 44.957 s versus baseline 12.225 s (3.677x), at
3 threads and FP32 on the full validation set. Peak working set was 1.669 GiB;
the complete uncompressed inference bundle is 54.131 MiB. All three measured
limits pass. No test-set score or test-set timing is claimed. Fourteen
correctness checks passed, including causality and window independence.

The inference bundle is `runs/hybrid-selected/`, containing just `checkpoint.pt`,
`hybrid.py`, `student.py`, and its short README. Every fitted table is inside
the checkpoint. Submit this one bundle rather than all experimental runs or
training-state files. Use the unchanged evaluator, tokenizer, and data:

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/hybrid-selected/checkpoint.pt --split validation --device cpu --precision fp32 --threads 3
```

For reproducible training-statistics fitting and finite validation selection:

```powershell
.\.venv\Scripts\python.exe fit_hybrid.py --source runs/rope-v4b-continuation-resumed/checkpoint.pt --run-dir runs/hybrid-v1 --threads 3
.\.venv\Scripts\python.exe refine_hybrid.py --source runs/hybrid-v1/checkpoint.pt --run-dir runs/hybrid-v2-long --threads 3
.\.venv\Scripts\python.exe fit_copy.py --source runs/hybrid-v2-long/checkpoint.pt --run-dir runs/hybrid-v3-copy --threads 3
.\.venv\Scripts\python.exe finalize_hybrid.py
```

Existing output directories are rejected; choose fresh names when repeating
experiments. `finalize_hybrid.py` records fixed-coefficient mechanism ablations
and alternates baseline/candidate CPU FP32 validation three times at three
threads. It refuses to overwrite completed results. No test scoring is performed
by these experiment scripts. The original evaluator's data loader still loads
all supplied splits internally, while the explicitly selected split is validation.

2–5 gram smoothing follows interpolated Modified Kneser–Ney; 6–10 gram repeated
phrases use discounted interpolation above that distribution. Long contexts
use a 64-bit lookup hash followed by exact token comparison, with conflicting
training hashes discarded. Neural cache entries pair a past hidden vector
at position `s` with already observed `x[s+1]` and require `s < t` when predicting
at `t`. Exact phrase copying has the same restriction. These mechanisms never
receive the evaluator's next-token target tensor.

References: [Chen & Goodman (1999)](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)
and [Grave et al. (2017)](https://arxiv.org/abs/1612.04426). The cache is adapted
to independent 256-token windows; it does not claim the paper's unrestricted
cache performance. Codex substantially assisted with the implementation,
correctness checks, bounded experiments, resource measurements, and report.

## 2026-09-28 continuation experiment

The current trainer saves the best full-validation model to `checkpoint.pt`,
the best ordinary/EMA weights to `best_raw.pt` / `best_ema.pt`, the latest
weights to `last_raw.pt` / `last_ema.pt`, and restart state to
`training_state.pt`. Only the selected inference checkpoint and its model
implementation are inference assets; optimizer, RNG and alternative weights
are training artifacts. Existing run directories are never overwritten.

`--init-from` starts a new optimizer stage and counts ancestor training
targets. `--resume` restores optimizer, sampling RNG, PyTorch CPU/CUDA RNG,
Python/NumPy RNG, EMA, validation selection and scheduler position. Resume
requires the same recipe flags and a **new output directory**. `--steps` is
the total stage horizon, not additional steps after resume. A legacy checkpoint
without optimizer state must use `--init-from`; it cannot exactly resume.
`--stop-after` permits controlled interruption while preserving this horizon.

The new CLI exposes `--lr`, `--min-lr`, `--warmup`, `--steps`, `--dropout`,
`--ema-decay`, and `--patience`. The cosine schedule reaches its minimum on
the last update; warmup scales initial updates. `--ema-decay 0` disables EMA.
Student dropout defaults to the original 0.1, now overridable by config/CLI.
EMA averages train-derived parameters after updates; it is scored separately
in FP32, and the selected EMA model has the same inference architecture.
The ancestor is evaluated and remains eligible as the best checkpoint.
Patience counts full validation events at which neither ordinary nor EMA
weights improves the global best BPB. Model train/eval mode is restored by
the unchanged evaluator, so residual dropout stays active while training.

The original bounded continuation command (PowerShell, existing environment) was:

```powershell
.\.venv\Scripts\python.exe train.py --init-from runs/rope-v4b/checkpoint.pt --run-dir runs/rope-v4b-continuation --steps 2000 --batch-size 32 --lr 1e-4 --min-lr 1e-5 --warmup 0 --ema-decay 0.99 --eval-every 400 --patience 2 --threads 3 --device cpu --precision fp32 --seed 23
```

A server restart terminated that process after the step-640 log. The step-400
training state and its best model survived. To count discarded updates within
the original 2000-update limit, the resumed stage explicitly replans its cosine
endpoint to step 1600 (`--resume-horizon 1600`). Optimizer, RNG, EMA and selection
are restored; this deliberate shorter schedule is recorded, not called an exact
continuation of the original schedule. Without `--resume-horizon`, restoration
keeps the saved schedule unchanged. `--checkpoint-every 100` now saves restart
state between validation events. The recovered experiment command is:

```powershell
.\.venv\Scripts\python.exe train.py --resume runs/rope-v4b-continuation/training_state.pt --resume-horizon 1600 --run-dir runs/rope-v4b-continuation-resumed --steps 2000 --batch-size 32 --lr 1e-4 --min-lr 1e-5 --warmup 0 --ema-decay 0.99 --eval-every 400 --patience 2 --checkpoint-every 100 --threads 3 --device cpu --precision fp32 --seed 23
.\.venv\Scripts\python.exe measure_eval.py --checkpoint runs/rope-v4b-continuation-resumed/checkpoint.pt --split validation --device cpu --precision fp32 --threads 3 --output runs/continuation_audit/candidate_validation.json
```

To reproduce training in fresh directories, run the original command with
`--stop-after 400`, then the recovered command pointing to that new directory.
Use new output names for both commands. Discarded work after step 400 has no
effect on restored weights but is still included in the cost audit. Its known
240–260 discarded updates plus the 1600-step trajectory and 2-step smoke run
total at most 1862 updates. See `runs/continuation_audit/recovery.json`.

`measure_eval.py` delegates scoring to the unchanged `evaluate.py` and adds
Windows native process peak working-set and committed-memory measurements.
Use the identical flags with the baseline checkpoint for resource comparisons.
No test evaluation is part of this experiment. The unchanged `load_data()`
internally loads all supplied splits; training uses only its train entry and
selection/scoring uses only its validation entry.

Original code copies, hashes, command logs and experiment evidence are under
`runs/continuation_audit/`. Engineering tests use tiny synthetic sequences only
for correctness; those models are temporary and never feed experiment weights.

AI assistance disclosure: OpenAI Codex substantially assisted with trainer
implementation, dropout configuration, restart/EMA tests, experiment execution,
resource measurement and documentation for this continuation experiment.

Read [the project guide](../guide/GUIDE.md) for the assignment, assessment, deadlines and peer review. This README contains the running instructions and technical rules. The package has only these two documents.

All commands below run from **code/**. Data and the tokenizer are included. No API key, pretrained weights or additional dataset download is needed; after installing dependencies, training and evaluation work offline.

## 1. Install

Use **Python 3.12**. From the extracted package directory:

```bash
cd code
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install PyTorch for **one** device:

```bash
# Linux/Windows CPU: recommended; no GPU needed
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a compatible driver, use this command **instead**:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

For macOS, install `torch==2.7.1` from the default PyPI index and run on CPU. After installing PyTorch, install the remaining dependencies and check the model:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Linux CPU commands were verified with Python 3.12 and PyTorch 2.7.1+cpu. Windows/macOS timings have not been measured.

## 2. Train and evaluate

**Quick installation check** — 10 training steps, then full-test evaluation:

```bash
python train.py --implementation model --steps 10 --run-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/checkpoint.pt --split test
```

This checks that the pipeline works; its score is **not** the full baseline. Each training run needs a new output directory.

**Full baseline** — 1,200 training updates, then evaluation:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

The baseline has four GPT blocks, width 128, four attention heads and **1,088,256 parameters**, and achieves approximately **2.10 test BPB**. On the reference four-thread Xeon Platinum 8457C, measured training took about **311 seconds** and scoring **5.92 seconds**, excluding installation and loading. These are reference measurements, not laptop guarantees or a fixed time allowance.

**Your model** — edit `student.py` and supporting files, then:

```bash
python train.py --implementation student --seed 17 --eval-every 300 --run-dir runs/my-model
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split validation
# Freeze the final method before testing:
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split test
```

Training writes `checkpoint.pt` and `metrics.json`. Evaluation writes `test_cpu_fp32.json` (or the corresponding device/split name) and per-window losses. Submit the **bpb** value from the complete-test JSON, not token perplexity or validation BPB. Default evaluation is FP32. Add `--device cuda` for GPU runs; training can use BF16, but ranked evaluation must use FP32 and remain reproducible on CPU. The supplied CUDA runner caps PyTorch allocation at 20 GB; driver overhead is additional.

## 3. Files and model interface

| Files | Use |
|---|---|
| `model.py`, `configs/baseline.json` | Runnable baseline; preserve for comparisons. |
| `student.py`, `train.py` | Your model factory and training recipe; add supporting code as needed. |
| `common.py`, `evaluate.py` | Fixed data checks, windows and scorer; keep unchanged. |
| `data/` | Supplied splits, tokenizer and dataset hashes; keep unchanged. |
| `tests/test_contract.py` | Checks your model's causality, normalization, independence and gradients. |
| `RUN_LOG_TEMPLATE.csv` | Optional experiment-log template. |
| `PACKAGE_MANIFEST.json` | Release hashes; paths are relative to the package root containing code/ and guide/. |

- `build_model(config)` returns a PyTorch model with `context=256`.
- The supplied trainer calls `forward(ids)` for unnormalized logits; the scorer calls `predict_log_probs(ids)` for finite, normalized natural-log probabilities. Both outputs have shape `[batch, time, 2048]`.
- A prediction at position t may use only the observed prefix through t. Reset temporary state between independent windows, examples and scoring passes. Compact training-derived assets may be reused across windows; evaluation-prefix state may not.
- Checkpoints record the implementation module and configuration. Include that module and every required asset so the evaluator can reconstruct the submitted predictor. No optimizer state is required for direct evaluation.
- Training length, architecture, optimizer, regularization, self-trained weight averaging and ensembles may change within the guide's constraints. Log all seeds, processed training targets, checkpoint ancestry and search costs; reusing a checkpoint does not erase its training cost. No particular seed or score improvement is mandated.

## 4. Benchmark and resource measurements

**Fixed score.** Protocol `7506-mp1-wt2-v2`: WikiText-2 raw text, train-fitted BPE-2048, independent windows of 256 targets, including the final short window. Every target except the first token of each split is scored once. Input windows share a boundary token but carry no state. BPB is summed negative log-base-2 next-token probability divided by the split's entire raw UTF-8 byte length, including the first token's bytes.

| Split | Scored targets | UTF-8 bytes |
|---|---:|---:|
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

Use validation for all development and checkpoint/mixture selection. Weights, statistics and retrieval entries must derive only from training text. The public test text enables reproduction; it must not be used to tune the method. Once frozen, the same predictor may be evaluated repeatedly for timing or reproduction. Token perplexity is not directly comparable with published word-level perplexity.

Measure all three limits for the same frozen predictor:

- **CPU time ≤5× baseline:**
- **Peak RAM ≤4 GiB:**
- **Inference assets ≤64 MiB uncompressed:** 

## 5. Prepare your submission and reproduce a peer

The [guide](../guide/GUIDE.md) specifies the deadline and website workflow. Include the following in your immutable code repository:

- **Report, at most 10 pages including figures, tables and references** 
- **Reproduction instructions**

Your final website submission must link to this code and the matching complete checkpoint bundle. The website generates the Issue JSON automatically. Keep all inference assets downloadable for verification.

To check a peer, obtain their exact code version and checkpoint, follow their installation instructions, and run their frozen model with the supplied evaluator:

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

Compare reproduced BPB with the reported score. Submit **Peer Review Report** with the reproduced score; optionally include the command, environment, difference and evidence/log link.  The instructor adjudicates discrepancies. Confirmed discrepancies during the seven-day review earn bonus credit under the announced marking policy.

## 6. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in `data/manifest.json`. These dataset notices do not assign a new license to the surrounding classroom code.
