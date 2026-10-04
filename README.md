# audio-eval-suite

**A fast, dependency-light evaluation suite for reverberation-robustness and
blind room-acoustic-parameter estimation.**

[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-audio--eval--suite-yellow)](https://huggingface.co/datasets/mandipgoswami/audio-eval-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data: CC-BY-4.0](https://img.shields.io/badge/data-CC--BY--4.0-blue.svg)](https://huggingface.co/datasets/mandipgoswami/audio-eval-suite)

This repo is the evaluation code for the
[`mandipgoswami/audio-eval-suite`](https://huggingface.co/datasets/mandipgoswami/audio-eval-suite)
dataset: 1,500 audio clips (300 source items × 5 conditions — clean, reverb,
noisy@10/5/0 dB) with ground-truth acoustic labels, built for two reference
tasks:

| Task | Type | Metric | Target |
|---|---|---|---|
| **Reverb-level classification** | 3-way | top-1 accuracy | `reverb_bin` ∈ {clean, mild, strong} |
| **RT60 regression** | regression | MAE / RMSE (seconds) | `t60_s` (anechoic `clean` excluded) |

It is deliberately small and instant-loading — a sanity/robustness check you can
drop into any pipeline, not a training corpus.

## Install

```bash
git clone https://github.com/mandip42/audio_eval_suite
cd audio_eval_suite
pip install -r requirements.txt
```

Dependencies are light: `numpy`, `scipy`, `soundfile`, `datasets`, `pandas`.
No librosa, no deep-learning stack.

## Quickstart

```bash
# Score the live Hub dataset (downloads it once) with the reference baseline:
python src/audio_eval_suite/run_eval.py

# Trivial label-free floors any real model must beat:
python src/audio_eval_suite/run_eval.py --baseline mean      # RT60: global-mean predictor
python src/audio_eval_suite/run_eval.py --baseline majority  # reverb-cls: majority-bin predictor

# Quick smoke run on the first 50 rows:
python src/audio_eval_suite/run_eval.py --limit 50
```

A one-cell version is in [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb)
(also runnable in Colab).

## Reference baselines

Fully reproducible floors shipped with the suite — a real estimator should clear
these comfortably:

| baseline | reverb-cls accuracy | RT60 MAE (s) | RT60 RMSE (s) |
|---|---|---|---|
| `signal` (direct-to-late energy ratio / Schroeder) | **0.4000** | **3.8939** | 4.8720 |
| `mean` / `majority` floor | 0.5833 | 1.7629 | 2.0704 |

(RT60 scored on n=1,200 — the 300 anechoic `clean` rows are excluded.)

The `signal` baseline predicts reverb class from an early-vs-late energy ratio
and estimates RT60 by classic Schroeder backward integration (T20→T60). Because
the clips are short synthetic speech-like probes rather than clean impulse
responses, the naive Schroeder estimate is intentionally noisy — these are
*floors*, documented as such.

## Submission format

To score your own model, produce one JSON file keyed by `f"{id}|{condition}"`:

```json
{
  "task_A": { "aes_0000|reverb": "mild", "aes_0000|noisy_snr10": "mild" },
  "task_B": { "aes_0000|reverb": 0.61,   "aes_0000|noisy_snr10": 0.59 }
}
```

- **task_A** ∈ {`clean`, `mild`, `strong`} for every row → top-1 accuracy.
- **task_B** float RT60 (seconds) for every non-clean row → MAE / RMSE.

Scoring reuses the metric functions in `run_eval.py` directly. See
[`EVAL_USAGE.md`](EVAL_USAGE.md) for the full spec.

## lm-evaluation-harness

[`tasks/audio_eval_suite.yaml`](tasks/audio_eval_suite.yaml) is an
lm-evaluation-harness-style task template (group `audio_eval_suite` →
`audio_eval_suite_rt60`, `audio_eval_suite_reverbcls`). Drop it into an
audio-capable harness fork's `tasks/` directory, or use it as the basis for an
upstream contribution.

## Related

Part of a room-acoustics research program:

- [RIR-Bench-Hard](https://huggingface.co/datasets/mandipgoswami/RIR-Bench-Hard) — adversarial RIR benchmark
- [reverb-speech-mini](https://huggingface.co/datasets/mandipgoswami/reverb-speech-mini) — dry/wet dereverberation pairs
- [rirmega](https://huggingface.co/datasets/mandipgoswami/rirmega) · [rir-mega-speech](https://huggingface.co/datasets/mandipgoswami/rir-mega-speech) · [rirmega-eval](https://huggingface.co/datasets/mandipgoswami/rirmega-eval)

## Citation

```bibtex
@misc{goswami2026audioevalsuite,
  title        = {audio-eval-suite: A Reverberation-Robustness and Blind RT60-Estimation Evaluation Suite},
  author       = {Goswami, Mandip},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/mandipgoswami/audio-eval-suite}}
}
```

## License

Code: MIT (see [LICENSE](LICENSE)). Dataset: CC-BY-4.0.
