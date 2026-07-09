# Energy Demand Forecasting — Foundation Models

Saatlik elektrik talebi (Monash `electricity_hourly`: 321 seri × 3 yıl) üzerinde uçtan uca
benchmark: klasik modeller, LightGBM, PatchTST ve Chronos-Bolt (zero-shot + fine-tuned), tek donuk
değerlendirme protokolüyle (5 rolling-origin haftalık pencere; MASE birincil, WQL + coverage@80
olasılıksal).

**Sonuç:** Foundation model kazandı. `chronos_bolt_base_ft` MASE **0.9358** (zero-shot base 0.9378)
ile SeasonalNaive(168) barajının (1.1130) %16 altında — hiçbir klasik model ve sıfırdan eğitilen
PatchTST (1.7116) barajı geçemedi. Tam tablo: `outputs/metrics/leaderboard.csv`.

## Dizin

| Yol | İçerik |
|---|---|
| `energy_forecasting_foundation_models.ipynb` | **Final notebook** — çalıştırılmış, çıktılı; tüm pipeline'ı tek başına yeniden üretir (Colab GPU, *Run all*) |
| `energy_forecasting_eda.ipynb` | **EDA notebook'u** — benchmark'taki her tasarım kararının veri temeli; bağımsızdır, veriyi Zenodo'dan kendisi indirir (Colab, GPU gerekmez) |
| `data/` | Girdi veri seti: ham `electricity_hourly_dataset.tsf` + notebook'un `CUSTOM_DATA` ile doğrudan okuyabileceği `electricity_hourly_long.parquet` (`unique_id, ds, y`) |
| `outputs/` | Final çıktılar: `REPORT.md` (**model karşılaştırma istatistikleri ve yorumları**), `metrics/` (leaderboard + model başına tablolar), `forecasts/` (kantil tahminleri, parquet), `models/` (Colab'de eğitilmiş final modeller — `chronos_bolt_base_ft/` HF'ye yüklenecek paylaşılabilir model: safetensors bf16 + model kartı), `train_logs/`, `run_meta.json` |
| `archive/` | Proje tarihçesi: geliştirme notebook'ları (EDA → harness → baseline'lar → PatchTST), Colab pipeline araçları, eski koşu sonuçları, ROADMAP, ham çıktı zip'i |

## Yeniden üretme

Notebook'u Colab'da (GPU, tercihen A100) açıp *Run all* — veri Zenodo'dan kendiliğinden iner,
~1,5–2 saatte 10 satırlık leaderboard'u ve paylaşılabilir modeli yeniden üretir. Başka bir veri
setiyle koşmak için config hücresinde `CUSTOM_DATA`'ya `unique_id, ds, y` kolonlu CSV/Parquet
gösterin (örn. `data/electricity_hourly_long.parquet` bu formattadır).

## Yayın

- **Model → Hugging Face:** `outputs/models/chronos_bolt_base_ft/` klasörünü olduğu gibi yükleyin
  (410 MB bf16 safetensors + config + model kartı; kart, benchmark sayılarını ve kullanım kodunu içerir).
- **Notebook → Kaggle:** kökteki çalıştırılmış benchmark `.ipynb`'sini import edin; EDA
  notebook'u da ayrı bir Kaggle notebook'u olarak yüklenebilir (anlatı sırası: önce EDA, sonra benchmark).
