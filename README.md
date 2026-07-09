# Energy Demand Forecasting — Foundation Models vs. Strong Baselines

An end-to-end benchmark on hourly electricity demand: classical statistical models, gradient
boosting, a from-scratch Transformer (PatchTST) and the **Chronos-Bolt time-series foundation
model** (zero-shot and fine-tuned), all evaluated under one frozen protocol — plus a shareable
fine-tuned model in Hugging Face format.

**Published model:** [`mskayacioglu/chronos-bolt-base-monash-electricity-hourly`](https://huggingface.co/mskayacioglu/chronos-bolt-base-monash-electricity-hourly)

**Kaggle notebooks:** [EDA](https://www.kaggle.com/code/mskayacioglu/energy-demand-forecasting-eda) · [Benchmark](https://www.kaggle.com/code/mskayacioglu/energy-demand-forecasting-with-foundation-models)

**Dataset:** Monash Time Series Forecasting Repository, `electricity_hourly` — 321 hourly series
of electricity consumption (kW), 2012–2014 (an aggregated version of UCI
*ElectricityLoadDiagrams20112014*). 26,304 timestamps per series, complete and equal-length.

## Results

Every model forecasts the next 168 hours (one week) for all 321 series jointly, over 5
non-overlapping rolling-origin windows. Point accuracy: **MASE** (seasonal m=24, primary),
sMAPE alongside. Probabilistic quality: **Weighted Quantile Loss** over nine quantiles (q10–q90)
and 80%-interval coverage. The bar to beat: **repeat last week** (seasonal naive, m=168).

| model | MASE | sMAPE | WQL | coverage@80 | vs. bar |
|---|---|---|---|---|---|
| **chronos_bolt_base_ft** | **0.9358** | 10.83 | **0.0896** | 0.701 | **−15.9%** |
| chronos_bolt_base (zero-shot) | 0.9378 | 10.68 | 0.0900 | 0.712 | −15.7% |
| chronos_bolt_small (zero-shot) | 0.9754 | 11.11 | 0.0920 | 0.710 | −12.4% |
| seasonal_naive_168 *(bar)* | 1.1130 | 11.71 | 0.1279 | 0.095 | — |
| lgbm_global | 1.2073 | 14.40 | 0.2828 | 0.974 | +8.5% |
| mstl | 1.2984 | 17.01 | 0.1398 | 0.884 | +16.7% |
| seasonal_naive_24 | 1.2993 | 14.33 | 0.1527 | 0.045 | +16.7% |
| auto_ets | 1.3988 | 17.40 | 0.2335 | 0.808 | +25.7% |
| patchtst | 1.7116 | 20.10 | 0.1517 | 0.721 | +53.8% |
| auto_theta | 1.7528 | 19.93 | 0.2337 | 0.956 | +57.5% |

Key findings (full statistics and interpretation in [`outputs/REPORT.md`](outputs/REPORT.md)):

- **Only the foundation-model family clears the seasonal-naive bar.** All three Chronos-Bolt
  variants reach MASE < 1; no classical model, global LightGBM or from-scratch Transformer does.
  The probabilistic gap is even larger: Chronos WQL (~0.090) is ~36% below the best
  well-calibrated classical alternative (MSTL, 0.140).
- **Chronos wins broadly, not on average tricks:** it beats the seasonal naive on 78% of all
  series×window pairs, uniformly across small/mid/large-scale series, and leads in every one of
  the five test windows.
- **Fine-tuning matches zero-shot on typical weeks and pays off exactly on the hardest week.**
  On the anomalous Christmas window its MASE improves from 1.317 to 1.282, better on 58% of the
  321 series (sign test p≈0.002) — robustness where forecasts matter most. This rests on the
  single holiday episode present in the test period.
- **Calibration caveat:** Chronos 80% intervals cover ~70–71% — slightly narrow. Apply a conformal
  correction if guaranteed coverage is required.

## Repository contents

| Path | Contents |
|---|---|
| `energy_forecasting_foundation_models.ipynb` | **Benchmark notebook** (executed, with outputs) — reproduces the entire pipeline in one run: data download, evaluation protocol, all 10 models, leaderboard, and the shareable fine-tuned model |
| `energy_forecasting_eda.ipynb` | **EDA notebook** — the data analysis behind every design decision in the benchmark (scale heterogeneity, seasonalities, anomalies, cross-series structure); standalone, CPU-only |
| `data/` | Versioned input dataset: raw `electricity_hourly_dataset.tsf`. The derived `electricity_hourly_long.parquet` is regenerable/local and intentionally not tracked |
| `outputs/REPORT.md` | Model-comparison statistics and interpretation |
| `outputs/metrics/` | `leaderboard.csv` + per-model metric tables (overall, per window, per series×window) |
| `outputs/forecasts/` | Generated quantile forecast parquet files; intentionally not tracked because they are large and reproducible |
| `outputs/models/` | Generated model artifacts; intentionally not tracked. The final fine-tuned model is published on Hugging Face |
| `outputs/train_logs/` | Generated training/validation curves; intentionally not tracked |
| `outputs/run_meta.json` | Tracked environment record for the benchmark run (A100, library versions, stop steps) |
| `archive/` | Superseded working files (not tracked) |

## Reproducing

Open the benchmark notebook in Colab (GPU runtime, A100 recommended) and *Run all* — the dataset
downloads itself from Zenodo; ~1.5–2 hours end-to-end regenerates the leaderboard and the
shareable model. The EDA notebook runs on any runtime (CPU is fine) in a few minutes.

The notebooks are also published on Kaggle:

- [Energy Demand Forecasting EDA](https://www.kaggle.com/code/mskayacioglu/energy-demand-forecasting-eda)
- [Energy Demand Forecasting with Foundation Models](https://www.kaggle.com/code/mskayacioglu/energy-demand-forecasting-with-foundation-models)

**Using your own data:** point `CUSTOM_DATA` in the benchmark notebook's configuration cell at any
CSV/Parquet with columns `unique_id, ds, y` on a regular time grid. The notebook can regenerate
`data/electricity_hourly_long.parquet` locally in this schema from the raw `.tsf` file. Every
model, metric and plot adapts automatically.

## Published model

The final fine-tuned model is published at
[`mskayacioglu/chronos-bolt-base-monash-electricity-hourly`](https://huggingface.co/mskayacioglu/chronos-bolt-base-monash-electricity-hourly).
It is `amazon/chronos-bolt-base` fine-tuned on the Monash `electricity_hourly` dataset.

Feed it the most recent history of any hourly demand-like series (up to 2048 points, no
covariates needed); it returns nine-quantile probabilistic forecasts for the next 64 steps
(longer horizons by rolling). The Hugging Face model card carries the benchmark numbers, usage
code, intended use, limitations and citations.

Install the inference package with `pip install chronos-forecasting`.

```python
import torch
from chronos import BaseChronosPipeline

pipe = BaseChronosPipeline.from_pretrained(
    "mskayacioglu/chronos-bolt-base-monash-electricity-hourly",
    device_map="cuda",
    dtype=torch.bfloat16,
)
```

## Data source

Godahewa et al., *Monash Time Series Forecasting Archive* (NeurIPS 2021) —
[`electricity_hourly`](https://zenodo.org/records/4656140), derived from UCI
ElectricityLoadDiagrams20112014.

## References

- Chronos / Chronos-Bolt base model: [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
- Dataset archive paper: [Monash Time Series Forecasting Archive](https://arxiv.org/abs/2105.06643)
- Dataset archive: [`electricity_hourly` on Zenodo](https://zenodo.org/records/4656140)

## License

This repository's code and documentation are released under the Apache License 2.0. The source
dataset and base model remain subject to their upstream terms.
