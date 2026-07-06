"""Shared evaluation harness for the energy-demand forecasting project.

This module is the **single source of truth** for the frozen Phase-3 evaluation
contract (see ``notebooks/02_evaluation_framework.ipynb`` and ``ROADMAP.md``).
It is extracted verbatim from notebook 02 so that every model notebook
(SeasonalNaive, ETS/Theta/MSTL/ARIMA, LightGBM, PatchTST, Chronos) is scored
through the *exact same* windows and metric code — no metric is ever
re-implemented in a model notebook.

Design note — **import safe**: the orchestration functions
(:func:`run_backtest`, :func:`evaluate`, :func:`update_leaderboard`) take
``wide`` / ``windows`` / ``series`` / ``index`` / paths as explicit arguments
instead of reaching for notebook globals. Importing this module has no side
effects; nothing is read from or written to disk until you call a function.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Evaluation constants (kept in sync with ROADMAP.md; the single source of truth)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
HORIZON = 168                       # forecast 1 week ahead (Monash standard)
SEASONALITY = 24                    # MASE seasonal period m (daily); confirmed dominant in EDA §9
QUANTILE_LEVELS = [round(0.1 * i, 1) for i in range(1, 10)]   # 0.1 .. 0.9
POINT_Q = 0.5
N_WINDOWS = 5                       # rolling-origin cutoffs -> variance estimate
STEP = HORIZON                      # non-overlapping weekly windows
COVERAGE_LO, COVERAGE_HI = 0.1, 0.9  # 80% central prediction interval
QCOLS = {q: f"q{int(q * 100):02d}" for q in QUANTILE_LEVELS}   # 0.1 -> 'q10'
METRIC_COLS = ["MAE", "RMSE", "sMAPE", "MASE", "WQL", "coverage80"]


# ---------------------------------------------------------------------------
# Paths & data loaders
# ---------------------------------------------------------------------------
def find_project_root(marker: str = "data/electricity_hourly_dataset.tsf", start=None) -> Path:
    """Walk upward from ``start`` (or cwd) until ``marker`` is found."""
    here = Path(start) if start is not None else Path.cwd()
    for base in [here, *here.parents]:
        if (base / marker).exists():
            return base
    raise FileNotFoundError(f"Could not locate {marker} from {here}")


def get_paths(project_root) -> SimpleNamespace:
    """Return the canonical project paths, creating the writable dirs."""
    project_root = Path(project_root)
    paths = SimpleNamespace(
        ROOT=project_root,
        INTERIM=project_root / "data" / "interim",
        PROCESSED=project_root / "data" / "processed",
        FORECASTS=project_root / "results" / "forecasts",
        METRICS=project_root / "results" / "metrics",
        FIG_DIR=project_root / "reports" / "figures",
    )
    for d in (paths.PROCESSED, paths.FORECASTS, paths.METRICS, paths.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return paths


def load_wide(interim_dir) -> pd.DataFrame:
    """Load the tidy long-format Parquet and pivot to a wide (ds × series) matrix."""
    long = pd.read_parquet(Path(interim_dir) / "electricity_hourly_long.parquet")
    wide = (
        long.pivot(index="ds", columns="unique_id", values="y")
        .sort_index()
        .astype("float64")
    )
    wide.columns = wide.columns.astype(str)
    return wide


def load_windows(processed_dir) -> pd.DataFrame:
    """Load the pinned rolling-origin cutoffs shared by every model notebook."""
    return pd.read_parquet(Path(processed_dir) / "backtest_windows.parquet")


def make_windows(index, n_windows, horizon, step, seasonality: int = SEASONALITY) -> pd.DataFrame:
    """Build ``n_windows`` non-overlapping rolling-origin windows walking back from the end."""
    last_pos = len(index) - 1
    rows = []
    for k in range(n_windows):
        cutoff_pos = last_pos - horizon - k * step   # last train position for window k
        if cutoff_pos - seasonality < 0:
            raise ValueError("Not enough history for the requested windows.")
        rows.append({
            "window": k,
            "cutoff_pos": cutoff_pos,
            "cutoff": index[cutoff_pos],
            "first_forecast": index[cutoff_pos + 1],
            "last_forecast": index[cutoff_pos + horizon],
        })
    return pd.DataFrame(rows).sort_values("window", ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics (hand-rolled; validated on synthetic data in notebook 02 §5)
# ---------------------------------------------------------------------------
def mae(y, p):        return np.abs(y - p).mean(axis=0)
def rmse(y, p):       return np.sqrt(((y - p) ** 2).mean(axis=0))


def smape(y, p):
    den = np.abs(y) + np.abs(p)
    with np.errstate(invalid="ignore", divide="ignore"):   # 0/0 rows are handled by np.where below
        val = np.where(den == 0, 0.0, 2.0 * np.abs(p - y) / den)
    return val.mean(axis=0) * 100.0


def seasonal_naive_denom(train_wide, m):
    """In-sample seasonal-naive MAE per series: mean|y_t - y_{t-m}| over the training portion."""
    a = train_wide.to_numpy()
    return np.abs(a[m:] - a[:-m]).mean(axis=0)          # (n_series,)


def mase(y, p, denom):
    d = np.where(denom == 0, np.nan, denom)
    return mae(y, p) / d


def pinball(y, q_pred, q):
    e = y - q_pred
    return np.maximum(q * e, (q - 1) * e)


def wql(y, quant_preds, levels):
    """Mean weighted quantile loss per series; quant_preds: {level: (H,n) array}."""
    num = np.zeros(y.shape[1])
    for q in levels:
        num += 2.0 * pinball(y, quant_preds[q], q).sum(axis=0)
    denom = np.abs(y).sum(axis=0)
    denom = np.where(denom == 0, np.nan, denom)
    return (num / len(levels)) / denom


def coverage(y, lo, hi):
    return ((y >= lo) & (y <= hi)).mean(axis=0)


# ---------------------------------------------------------------------------
# Backtest orchestrator & forecast I/O
# ---------------------------------------------------------------------------
def run_backtest(model_fn, name, *, wide, windows, series, forecasts_dir,
                 horizon: int = HORIZON, quantile_levels=QUANTILE_LEVELS, save: bool = True):
    """Loop the shared cutoffs, call ``model_fn(train_wide, h)``, and assemble the
    tidy forecast frame in the contract schema.

    ``model_fn`` returns either an ``(h, n_series)`` array (point model, reused for
    every quantile) or a dict ``{quantile_level: (h, n_series) array}`` (probabilistic).
    Columns of every ``(h, n_series)`` array must follow ``series`` order.
    """
    series = pd.Index(series)
    n = len(series)
    qcols = {q: f"q{int(q * 100):02d}" for q in quantile_levels}
    uid = np.tile(series.to_numpy(), horizon)
    horizon_idx = np.repeat(np.arange(1, horizon + 1), n)
    blocks = []
    for _, w in windows.iterrows():
        pos = int(w["cutoff_pos"])
        train = wide.iloc[: pos + 1]
        y_true = wide.iloc[pos + 1 : pos + 1 + horizon].to_numpy()      # (H, n)
        ds = wide.index[pos + 1 : pos + 1 + horizon]

        preds = model_fn(train, horizon)
        if not isinstance(preds, dict):
            preds = {q: preds for q in quantile_levels}

        block = pd.DataFrame({
            "model": name,
            "unique_id": uid,
            "cutoff": w["cutoff"],
            "ds": np.repeat(ds.to_numpy(), n),
            "horizon": horizon_idx,
            "y": y_true.reshape(-1),
        })
        for q, col in qcols.items():
            block[col] = np.asarray(preds[q]).reshape(-1)
        blocks.append(block)

    fc = pd.concat(blocks, ignore_index=True)
    if save:
        forecasts_dir = Path(forecasts_dir)
        path = forecasts_dir / f"{name}.parquet"
        fc.to_parquet(path, index=False)
        print(f"[{name}] saved {len(fc):,} rows -> "
              f"{forecasts_dir.parent.name}/{forecasts_dir.name}/{name}.parquet")
    return fc


# ---------------------------------------------------------------------------
# Centralized evaluation & leaderboard
# ---------------------------------------------------------------------------
def evaluate(name, *, wide, index, series, forecasts_dir, metrics_dir,
             m: int = SEASONALITY, write: bool = True):
    """Score a forecast file identically for every model; returns (overall, per)."""
    series = pd.Index(series)
    forecasts_dir = Path(forecasts_dir)
    metrics_dir = Path(metrics_dir)

    def _pivot(g, col):
        return (
            g.pivot_table(index="horizon", columns="unique_id", values=col)
            .reindex(columns=series)
            .to_numpy()
        )

    fc = pd.read_parquet(forecasts_dir / f"{name}.parquet")
    per_window = []
    for cutoff, g in fc.groupby("cutoff", sort=True):
        pos = int(index.get_loc(cutoff))
        denom = seasonal_naive_denom(wide.iloc[: pos + 1], m)     # (n,)
        y = _pivot(g, "y")
        p50 = _pivot(g, QCOLS[POINT_Q])
        quant = {q: _pivot(g, QCOLS[q]) for q in QUANTILE_LEVELS}
        per_window.append(pd.DataFrame({
            "cutoff": cutoff,
            "unique_id": series,
            "MAE": mae(y, p50),
            "RMSE": rmse(y, p50),
            "sMAPE": smape(y, p50),
            "MASE": mase(y, p50, denom),
            "WQL": wql(y, quant, QUANTILE_LEVELS),
            "coverage80": coverage(y, quant[COVERAGE_LO], quant[COVERAGE_HI]),
        }))
    per = pd.concat(per_window, ignore_index=True)

    overall = per[METRIC_COLS].mean(numeric_only=True)          # mean over (series, window)
    overall["model"] = name
    overall["n_excluded_MASE"] = int(per["MASE"].isna().sum())
    if write:
        by_window = per.groupby("cutoff")[METRIC_COLS].mean()
        per.to_csv(metrics_dir / f"{name}_per_series.csv", index=False)
        by_window.to_csv(metrics_dir / f"{name}_by_window.csv")
        pd.DataFrame([overall]).to_csv(metrics_dir / f"{name}_summary.csv", index=False)
    return overall, per


def update_leaderboard(*overalls, metrics_dir):
    """Merge one or more ``overall`` rows into the shared leaderboard, sorted by MASE."""
    path = Path(metrics_dir) / "leaderboard.csv"
    cols = ["model", "MASE", "sMAPE", "RMSE", "MAE", "WQL", "coverage80", "n_excluded_MASE"]
    new = pd.DataFrame(list(overalls))[cols]
    if path.exists():
        board = pd.read_csv(path)
        board = board[~board["model"].isin(new["model"])]
        board = pd.concat([board, new], ignore_index=True)
    else:
        board = new
    board = board.sort_values("MASE", ignore_index=True)
    board.to_csv(path, index=False)
    return board
