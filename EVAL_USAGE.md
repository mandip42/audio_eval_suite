# audio-eval-suite — eval harness

Runnable reference evaluator for the two tasks that ship with
[`mandipgoswami/audio-eval-suite`](https://huggingface.co/datasets/mandipgoswami/audio-eval-suite):

- **Task A — reverb-level classification** → top-1 **accuracy** (label `reverb_bin` ∈ {clean, mild, strong})
- **Task B — RT60 regression** → **MAE** / RMSE in seconds vs `t60_s` (clean condition excluded — it is anechoic)

Everything here is dependency-light: `numpy`, `scipy`, `soundfile`, and
`datasets` (or `pandas` for the local path). No librosa, no deep-learning stack.

## Install

```bash
pip install datasets soundfile scipy numpy pandas
```

## Run

```bash
# Score the live Hub dataset (downloads it once):
python eval/run_eval.py

# Score a local staging copy (offline) — the folder with test.parquet + audio/:
python eval/run_eval.py --local .

# Trivial label-free floors (what any real model must beat):
python eval/run_eval.py --local . --baseline mean       # Task B: mean-t60 predictor
python eval/run_eval.py --local . --baseline majority    # Task A: majority-bin predictor

# Quick smoke run on the first 50 rows:
python eval/run_eval.py --local . --limit 50
```

It prints a small results table and writes **`eval/baseline_results.json`**.

## What the baselines actually do

| baseline | Task A (classification) | Task B (RT60) |
|---|---|---|
| `signal` *(default)* | predict clean/mild/strong from a **direct-to-late energy ratio** (early-window vs tail energy, dB) thresholded by fixed constants — label-free at inference | **Schroeder backward integration** of the clip energy, a line fit over the [-5, -25] dB decay window extrapolated to -60 dB (classic T20→T60) |
| `mean` / `majority` | predict the data-majority bin | predict the global-mean `t60_s` |

Both are honest, fully reproducible **floors**, not strong models. On this set the
clips are short synthetic speech-like probes (not clean impulse responses), so the
naive Schroeder estimator is noisy and the trivial mean predictor is a *lower* MAE
— that is expected and is exactly why these are floors a real estimator should
clear comfortably.

## Measured baseline numbers (local staging, 1500 rows)

| baseline | Task A accuracy | Task B MAE (s) | Task B RMSE (s) |
|---|---|---|---|
| `signal`   | **0.4000** | **3.8939** | 4.8720 |
| `mean/majority` | 0.5833 | 1.7629 | 2.0704 |

(`n=1200` for Task B — the 300 clean rows are excluded.)

## Submission format

To score your own model, produce **one JSON file** with per-row predictions keyed
by `id` + `condition` (the two together are the primary key — each `id` has 5
conditions):

```json
{
  "task_A": {
    "aes_0000|clean":  "clean",
    "aes_0000|reverb": "mild",
    "aes_0000|noisy_snr10": "mild",
    "...": "..."
  },
  "task_B": {
    "aes_0000|reverb": 0.61,
    "aes_0000|noisy_snr10": 0.59,
    "...": 0.0
  }
}
```

- **Task A** value ∈ {`clean`, `mild`, `strong`} for every row.
- **Task B** value is a float RT60 estimate in seconds for every **non-clean** row
  (clean rows may be omitted or ignored).
- Keys are `f"{id}|{condition}"`.

Scoring is identical to `run_eval.py`: `task_A` → top-1 accuracy over all rows;
`task_B` → MAE/RMSE over non-clean rows, overall and per condition.

A reference scorer for a submission file can reuse the metric functions in
`run_eval.py` (`_mae`, `_rmse`) directly.

## Files

- `run_eval.py` — the evaluator (both tasks, real + trivial baselines)
- `baseline_results.json` — committed output of the default `signal` baseline
- `lm_eval_task.yaml` — an lm-evaluation-harness-style task template (see its header)
