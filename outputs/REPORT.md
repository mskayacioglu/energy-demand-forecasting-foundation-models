# Model Comparison Report

Result analysis of the hourly electricity demand forecasting benchmark. All numbers come from the
frozen evaluation protocol (5 rolling-origin weekly windows × 168-hour horizon, 321 series); the
raw tables live under `outputs/metrics/` and each section cites its source. Run: Colab A100
(`run_meta.json`).

## 1. Overall ranking (`metrics/leaderboard.csv`)

| model | MASE | sMAPE | WQL | coverage@80 | vs. bar |
|---|---|---|---|---|---|
| **chronos_bolt_base_ft** | **0.9358** | 10.83 | **0.0896** | 0.701 | **−15.9%** |
| chronos_bolt_base | 0.9378 | 10.68 | 0.0900 | 0.712 | −15.7% |
| chronos_bolt_small | 0.9754 | 11.11 | 0.0920 | 0.710 | −12.4% |
| seasonal_naive_168 (bar) | 1.1130 | 11.71 | 0.1279 | 0.095 | — |
| lgbm_global | 1.2073 | 14.40 | 0.2828 | 0.974 | +8.5% |
| mstl | 1.2984 | 17.01 | 0.1398 | 0.884 | +16.7% |
| seasonal_naive_24 | 1.2993 | 14.33 | 0.1527 | 0.045 | +16.7% |
| auto_ets | 1.3988 | 17.40 | 0.2335 | 0.808 | +25.7% |
| patchtst | 1.7116 | 20.10 | 0.1517 | 0.721 | +53.8% |
| auto_theta | 1.7528 | 19.93 | 0.2337 | 0.956 | +57.5% |

**Headline finding:** only the foundation-model family clears the *repeat-last-week* bar. All
three Chronos-Bolt variants reach MASE < 1 (better even than the in-sample seasonal-naive
scaling); none of the classical per-series models, the global LightGBM, or the from-scratch
PatchTST does. The probabilistic gap is larger still: Chronos WQL (~0.090) is 36% below the best
well-calibrated classical alternative (MSTL, 0.140).

## 2. Window breakdown — the holiday effect (`metrics/*_by_window.csv`)

MASE by window cutoff:

| cutoff | snaive168 | lgbm | patchtst | chronos base | chronos ft |
|---|---|---|---|---|---|
| Nov 26 | 0.897 | 0.910 | 1.154 | 0.716 | 0.728 |
| Dec 03 | 1.009 | 1.051 | 1.232 | 0.812 | 0.812 |
| Dec 10 | 1.014 | 1.120 | 1.181 | 0.821 | 0.831 |
| Dec 17 | 1.080 | 1.197 | 1.517 | 1.024 | 1.026 |
| **Dec 24 (Christmas)** | **1.565** | **1.759** | **3.474** | **1.317** | **1.282** |

- The Christmas week is the hardest window for every model, but the separation between models
  turns into a chasm there: PatchTST collapses to 3.47 (no calendar awareness) while Chronos
  contains the damage at 1.28–1.32.
- Chronos leads in **every one** of the five windows — its advantage does not come from a single
  lucky week.

## 3. What fine-tuning contributes (ft vs. zero-shot base)

- The aggregate difference is marginal: MASE 0.9358 vs. 0.9378, WQL 0.0896 vs. 0.0900.
  Fine-tuning stopped early (350–650 steps per window, 400 for the final fit;
  `train_logs/ft_*_val_curve.csv`) — pretraining already covers this data pattern well.
- **Where the gain lives:** per-window MASE delta (ft − base) is +0.012, −0.001, +0.010, +0.002,
  **−0.034** — i.e. ft is level with or a hair behind base on the four ordinary weeks, and its one
  meaningful gain is on the hardest window, the Christmas week (1.317 → 1.282). Operationally that
  is a valuable property: extra robustness exactly when forecasts matter most.
- **The hard-week gain is statistically significant and broad-based:** on the Christmas window ft
  beats base on 58% of the 321 series (187/321; sign test p≈0.002; 57% on WQL), and a quarter of
  the series improve by more than 0.08 MASE. On ordinary windows the win share drops to 33–46%.
  The mechanism is consistent: the fine-tuning data contained the 2012–2013 Christmas weeks, so
  the model could learn this grid's own holiday behaviour. Caveat: this is the only anomalous
  episode in the test period — a general "better in exceptional periods" claim rests on a single
  holiday episode and should be made cautiously.
- Head-to-head across all series×window pairs, ft beats base only 44% of the time (46% on WQL):
  the correct claim is not "better everywhere" but **"same average, sturdier on the hard week."**

## 4. Breadth of the win & scale sensitivity (`metrics/*_per_series.csv`)

- ft beats the seasonal naive on **78%** of series×window pairs (base: 79%) — the advantage comes
  from the whole distribution, not a few large series. Against PatchTST the win share is 95%.
- By scale tercile (series grouped by mean level), ft's MASE is 0.924 / 0.949 / 0.934 for
  small / mid / large series — balanced, scale-independent. (Reference: seasonal naive
  1.107 / 1.138 / 1.094.)

## 5. Calibration (coverage@80, nominal 0.80)

| model | coverage@80 | reading |
|---|---|---|
| Chronos family | 0.70–0.71 | intervals ~9 points narrow → mildly overconfident tails |
| auto_ets | 0.808 | near-perfect calibration (but weak point accuracy) |
| mstl | 0.884 | slightly wide |
| lgbm / auto_theta | 0.97 / 0.96 | far too wide (this also inflates their WQL) |
| seasonal naives | 0.05–0.10 | degenerate (point models; no real intervals) |

**Recommendation:** if Chronos is served in production and coverage guarantees are required,
calibrate the intervals with a conformal correction (quantile widening from past residuals);
the model card carries the same caveat.

## 6. PatchTST: the trained-Transformer reference

PatchTST trained with a generous, early-stopped budget (training halted by validation loss at
1.6k–2.5k steps per window, far below the ceiling) and still does not approach the bar: 1.71
overall. The window breakdown shows why: 1.15–1.52 on the four ordinary weeks (close to the bar),
3.47 on the Christmas week. The architecture sees only past values — with no calendar input, the
holiday pattern is invisible to it. This mirrors the common literature finding that a
from-scratch Transformer struggles to beat a strong seasonal naive on data with dominant, stable
seasonality.

## 7. Conclusion & model selection

1. **Model to serve:** `chronos_bolt_base_ft` (`outputs/models/chronos_bolt_base_ft/`, bf16
   safetensors + model card). Rationale: leaderboard #1, most robust on the hardest window,
   scale-balanced. The zero-shot base is a sound fallback/reference (0.2% behind).
2. **Cheap alternative:** if nothing can be deployed, `seasonal_naive_168` remains a respectable
   default; every classical/ML model in this benchmark is worse than it.
3. **Open items:** quantile calibration (conformal), and — if desired — a calendar-aware variant
   (a covariate-capable model or a holiday-aware post-correction).
