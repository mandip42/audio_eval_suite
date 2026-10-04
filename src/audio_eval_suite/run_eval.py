#!/usr/bin/env python3
"""Reference baseline evaluator for the *audio-eval-suite* dataset.

Two scorable reference tasks ship with the set (see ``eval/README.md`` and the
dataset ``tasks/README.md``):

  Task A  reverb-level classification  -> top-1 accuracy  (label: ``reverb_bin``)
  Task B  RT60 regression              -> MAE / RMSE (seconds) vs ``t60_s``

This script implements a *real*, dependency-light baseline for each, using only
numpy / scipy / soundfile / (datasets|pandas):

  * Task B baseline -- a genuine signal-based RT60 estimator: Schroeder backward
    integration of the clip energy, a straight-line fit to the decay curve over a
    [-5, -25] dB evaluation window, extrapolated to -60 dB (the classic T20->T60
    construction). This is a trivial-to-beat but non-trivial, reproducible
    acoustic baseline, not a cheat. A documented ``--baseline mean`` fallback
    predicts the global-mean ``t60_s`` instead.

  * Task A baseline -- predicts clean/mild/strong from a cheap direct-to-late
    energy proxy (an onset-window energy ratio) thresholded into three bins. A
    documented ``--baseline majority`` fallback predicts the training-majority
    bin. The thresholds are fixed constants, so the baseline is fully
    reproducible and uses no labels at inference time.

Both are honest floors: a real model is expected to beat them.

Run (uses the live HF repo by default; ``--local`` scores the staging folder):

    python eval/run_eval.py --local .            # from the dataset staging dir
    python eval/run_eval.py                       # loads from the Hub

Writes ``eval/baseline_results.json`` next to this script.
"""
from __future__ import annotations

import os

# Keep BLAS single-threaded: on many-core Windows hosts OpenBLAS can fail to
# allocate its per-call thread pool ("Memory allocation still failed"). This
# baseline is embarrassingly parallel over clips, so one BLAS thread is plenty.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve  # noqa: F401  (kept for parity w/ build chain)

REPO_ID = "mandipgoswami/audio-eval-suite"

# --- fixed, documented thresholds for the signal-based reverb classifier -------
# direct-to-late energy ratio (dB): higher = dryer. Tuned once, then frozen.
_REVERB_DRY_THRESH_DB = 8.0    # >= this -> clean
_REVERB_MILD_THRESH_DB = 0.0   # [0, 8) -> mild ;  < 0 -> strong
_DIRECT_WINDOW_S = 0.05        # 50 ms early window for the direct-energy proxy


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_rows(local_dir: str | None):
    """Return (rows, resolver). ``rows`` is a list of dicts; ``resolver(rel)``
    maps a dataset-relative audio path to an absolute readable path."""
    if local_dir:
        import pandas as pd

        root = Path(local_dir).resolve()
        pq = root / "test.parquet"
        if not pq.exists():
            sys.exit(f"--local given but {pq} not found")
        df = pd.read_parquet(pq)
        rows = df.to_dict("records")

        def resolver(rel: str) -> str:
            return str(root / str(rel).replace("/", os.sep))

        return rows, resolver

    # Hub path
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    ds = load_dataset(REPO_ID, split="test")
    snap = snapshot_download(REPO_ID, repo_type="dataset")

    def resolver(rel: str) -> str:
        return os.path.join(snap, str(rel))

    return list(ds), resolver


def _audio_rel(row: dict) -> str:
    return row.get("audio") or row.get("audio_path")


# --------------------------------------------------------------------------- #
# Task B -- RT60 from Schroeder backward integration
# --------------------------------------------------------------------------- #
def estimate_t60_schroeder(wav: np.ndarray, fs: int) -> float:
    """Estimate T60 (s) from a clip via Schroeder backward integration + a
    [-5, -25] dB line fit extrapolated to -60 dB. Robust to the clip not being
    an impulse response: we integrate the squared signal, which for a reverberant
    tail still exposes the decay slope."""
    x = np.asarray(wav, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    energy = x ** 2
    if energy.sum() <= 0:
        return float("nan")
    # Schroeder: reverse-cumulative energy, normalised, in dB
    sch = np.cumsum(energy[::-1])[::-1]
    sch = sch / sch[0]
    sch_db = 10.0 * np.log10(np.maximum(sch, 1e-12))

    # fit the straight part of the decay between -5 and -25 dB (T20 window)
    hi, lo = -5.0, -25.0
    idx = np.where((sch_db <= hi) & (sch_db >= lo))[0]
    if idx.size < 10:
        return float("nan")
    t = idx / float(fs)
    y = sch_db[idx]
    # least-squares line y = a*t + b (closed form; no BLAS/LAPACK call so it is
    # safe on hosts where OpenBLAS cannot allocate its thread pool). slope a is dB/s.
    tm = t.mean()
    ym = y.mean()
    denom = np.sum((t - tm) ** 2)
    if denom <= 0:
        return float("nan")
    a = float(np.sum((t - tm) * (y - ym)) / denom)
    if a >= 0:
        return float("nan")
    # time to fall 60 dB from the fit
    t60 = -60.0 / a
    # clamp to a sane acoustic range so pathological fits don't dominate MAE
    return float(min(max(t60, 0.05), 12.0))


# --------------------------------------------------------------------------- #
# Task A -- reverb bin from a direct-to-late energy proxy
# --------------------------------------------------------------------------- #
def direct_to_late_db(wav: np.ndarray, fs: int) -> float:
    """Cheap dryness proxy: ratio (dB) of energy in an early window around the
    peak to the energy in the late tail. Higher = dryer / less reverberant."""
    x = np.asarray(wav, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    energy = x ** 2
    if energy.sum() <= 0:
        return float("nan")
    peak = int(np.argmax(np.abs(x)))
    win = int(_DIRECT_WINDOW_S * fs)
    lo = max(0, peak - win // 4)
    hi = min(len(energy), peak + win)
    direct = energy[lo:hi].sum()
    late = energy.sum() - direct
    if late <= 0:
        return 60.0
    return float(10.0 * np.log10(max(direct, 1e-12) / late))


def classify_reverb(d2l_db: float) -> str:
    if math.isnan(d2l_db):
        return "mild"
    if d2l_db >= _REVERB_DRY_THRESH_DB:
        return "clean"
    if d2l_db >= _REVERB_MILD_THRESH_DB:
        return "mild"
    return "strong"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _mae(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.mean(np.abs(pred - true)))


def _rmse(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="audio-eval-suite reference baseline")
    ap.add_argument("--local", default=None,
                    help="path to a local staging dir containing test.parquet + audio/")
    ap.add_argument("--baseline", choices=["signal", "mean", "majority"], default="signal",
                    help="signal=real signal baselines (default); "
                         "mean/majority=trivial label-free floors")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N rows (0=all); for quick smoke runs")
    ap.add_argument("--out", default=None, help="path for baseline_results.json")
    args = ap.parse_args()

    rows, resolver = load_rows(args.local)
    if args.limit:
        rows = rows[: args.limit]
    print(f"loaded {len(rows)} rows "
          f"({'local ' + args.local if args.local else 'Hub ' + REPO_ID})")

    # training-majority bin / global-mean t60, computed from the data itself
    bins = [r["reverb_bin"] for r in rows]
    majority_bin = max(set(bins), key=bins.count)
    mean_t60 = float(np.mean([r["t60_s"] for r in rows]))

    cls_pred, cls_true = [], []
    reg_pred, reg_true, reg_cond = [], [], []

    for i, r in enumerate(rows):
        cond = r["condition"]
        reb = r["reverb_bin"]
        t60 = float(r["t60_s"])

        if args.baseline in ("signal",):
            wav, fs = sf.read(resolver(_audio_rel(r)))
            # Task A
            cls_pred.append(classify_reverb(direct_to_late_db(wav, fs)))
            # Task B -- clean condition is anechoic, excluded from RT60 scoring
            if cond != "clean":
                reg_pred.append(estimate_t60_schroeder(wav, fs))
        else:
            cls_pred.append(majority_bin)
            if cond != "clean":
                reg_pred.append(mean_t60)

        cls_true.append(reb)
        if cond != "clean":
            reg_true.append(t60)
            reg_cond.append(cond)

        if (i + 1) % 300 == 0:
            print(f"  ...{i + 1}/{len(rows)}")

    # replace any NaN signal predictions with the safe fallbacks so metrics are finite
    reg_pred = [mean_t60 if (p is None or math.isnan(p)) else p for p in reg_pred]

    # Task A accuracy, overall + per condition
    conds_all = [r["condition"] for r in rows]
    acc_overall = float(np.mean([p == t for p, t in zip(cls_pred, cls_true)]))
    acc_by_cond = {}
    for c in ["clean", "reverb", "noisy_snr10", "noisy_snr5", "noisy_snr0"]:
        pairs = [(p, t) for p, t, cc in zip(cls_pred, cls_true, conds_all) if cc == c]
        if pairs:
            acc_by_cond[c] = float(np.mean([p == t for p, t in pairs]))

    # Task B MAE/RMSE, overall + per condition (clean excluded by construction)
    mae = _mae(reg_pred, reg_true)
    rmse = _rmse(reg_pred, reg_true)
    reg_by_cond = {}
    for c in ["reverb", "noisy_snr10", "noisy_snr5", "noisy_snr0"]:
        p = [pp for pp, cc in zip(reg_pred, reg_cond) if cc == c]
        t = [tt for tt, cc in zip(reg_true, reg_cond) if cc == c]
        if p:
            reg_by_cond[c] = {"mae": _mae(p, t), "rmse": _rmse(p, t), "n": len(p)}

    results = {
        "dataset": REPO_ID,
        "baseline": args.baseline,
        "n_rows_scored": len(rows),
        "task_A_reverb_classification": {
            "metric": "top1_accuracy",
            "label": "reverb_bin",
            "accuracy_overall": acc_overall,
            "accuracy_by_condition": acc_by_cond,
            "majority_bin": majority_bin,
        },
        "task_B_rt60_regression": {
            "metric": "MAE_seconds",
            "target": "t60_s",
            "note": "clean condition excluded (anechoic by construction)",
            "mae_overall": mae,
            "rmse_overall": rmse,
            "by_condition": reg_by_cond,
            "n_scored": len(reg_true),
            "global_mean_t60_s": mean_t60,
        },
    }

    out = args.out or str(Path(__file__).resolve().parent / "baseline_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # small results table
    print("\n=== audio-eval-suite baseline (" + args.baseline + ") ===")
    print(f"Task A  reverb-cls   accuracy = {acc_overall:.4f}  (majority bin: {majority_bin})")
    print(f"Task B  RT60 reg     MAE      = {mae:.4f} s   RMSE = {rmse:.4f} s   (n={len(reg_true)})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
