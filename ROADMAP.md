# Enerji Talep Tahmini — Foundation Models · Yol Haritası (Roadmap)

Saatlik elektrik tüketim verisiyle gelecek enerji talebini tahmin eden; klasik
baseline'ları, PatchTST (transformer) ve Chronos (time-series foundation model)
karşılaştıran; **performanslı, deploy edilebilir ve ölçülebilir** bir sistem.

---

## 0. Proje Özeti ve Karar Sabitleri

Bu bölüm tüm fazların dayandığı ortak varsayımlardır. Değişirse buradan güncelle.

| Konu | Karar | Gerekçe |
|------|-------|---------|
| **Veri seti** | `data/electricity_hourly_dataset.tsf` (Monash TSF) | 321 saatlik seri, 2012–2014, 26.304 adım/seri, eksik değer yok |
| **Görev** | Multi-series (global) tek adımlı **çok-ufuklu** tahmin | Tüm seriler tek modelde; foundation modellerle uyumlu |
| **Tahmin ufku (H)** | **168 saat** (1 hafta) | Monash benchmark standardı — literatürle kıyaslanabilir |
| **Bağlam (context)** | 512–720 saat (deney değişkeni) | ~3–4 hafta geçmiş; günlük+haftalık mevsimsellik yakalar |
| **Mevsimsellik** | Günlük (24), haftalık (168) | Saatlik enerji talebinin baskın periyotları |
| **Birincil metrik** | **MASE** + **WQL/CRPS** | Nokta + olasılıksal; foundation modeller olasılıksal üretir |
| **Değerlendirme** | Rolling-origin (backtest), son pencereler test | Sızıntısız, gerçekçi |

**Başarı tanımı (proje sonu):** En iyi model, seasonal-naive baseline'ını MASE'de
anlamlı biçimde geçer; p50/p90 tahminleri kalibre; < X ms/seri gecikmeyle
container'da servis edilir; sonuçlar tek komutla üretilebilir (reproducible).

---

## Faz 1 — Altyapı ve Proje İskeleti  ·  (~2 gün)

**Amaç:** Deneylerin tekrarlanabilir, izlenebilir ve temiz olması.

- [ ] Ortam yönetimi: `uv` (veya `poetry`) + `pyproject.toml`, Python 3.11
- [ ] Repo yapısı (aşağıda), `src/` layout + editable install
- [ ] Konfig yönetimi: **Hydra** / OmegaConf (`configs/`) — model & veri parametreleri koddan ayrı
- [ ] Deney takibi: **MLflow** (veya W&B) — metrik, parametre, artefakt loglama
- [ ] Kod kalitesi: `ruff` + `black` + `mypy`, `pre-commit` hook'ları
- [ ] Test iskeleti: `pytest`; determinizm için global `seed` yardımcıları
- [ ] `Makefile` / task runner: `make data`, `make train`, `make eval`, `make serve`

**Önerilen dizin yapısı**
```
energy-demand-forecasting-foundation-models/
├── data/
│   ├── raw/            # electricity_hourly_dataset.tsf
│   ├── interim/        # parse edilmiş long-format parquet
│   └── processed/      # train/val/test pencereleri
├── configs/            # Hydra: data.yaml, model/*.yaml, eval.yaml
├── src/energyfc/
│   ├── data/           # tsf parser, split, dataset
│   ├── models/         # baselines, patchtst, chronos wrapper
│   ├── evaluation/     # metrikler, backtest, raporlama
│   ├── serving/        # FastAPI app, inference
│   └── utils/
├── notebooks/          # EDA, hata analizi (sadece keşif)
├── scripts/            # CLI giriş noktaları
├── tests/
└── ROADMAP.md
```

**DoD:** `make train` boş bir baseline'ı uçtan uca çalıştırıp MLflow'a run yazıyor.

---

## Faz 2 — Veri Anlama ve EDA  ·  (~2–3 gün)

**Amaç:** Verinin karakterini bilmeden model seçme.

- [ ] `.tsf` parser (Monash formatı): `series_name`, `start_timestamp`, değer dizisi → **long-format** (`unique_id`, `ds`, `y`) parquet'e yaz
- [ ] Doğrulama: 321 seri, seri başına 26.304 adım, tarih ekseni 2012–2014 sürekli mi?
- [ ] EDA:
  - [ ] Serilerin ölçek dağılımı (bazı müşteriler çok büyük → normalizasyon şart)
  - [ ] Günlük/haftalık mevsimsellik (ACF/PACF, STL decomposition)
  - [ ] Trend ve seviye kaymaları, tatil/anomali etkileri
  - [ ] Sıfır/sabit dönemler, aykırı değerler
  - [ ] Seriler arası benzerlik (kümeleme — global modelin işine yarar mı?)
- [ ] Zaman özellikleri: saat, haftanın günü, ay, tatil bayrağı (calendar features)

**DoD:** `notebooks/01_eda.ipynb` + veri kalitesi özeti; long-format parquet hazır.

---

## Faz 3 — Değerlendirme Çerçevesi (ÖNCE bunu kur)  ·  (~2–3 gün)

**Amaç:** Model yazmadan önce "adil karşılaştırma" mekanizmasını sabitlemek —
tüm modeller aynı split, aynı metrik, aynı backtest ile ölçülür.

- [ ] **Split stratejisi:** kronolojik. Son 168 saat = test; ondan önceki 168 = validation. Geri kalan = train
- [ ] **Rolling-origin backtest:** birden çok cutoff (örn. 3–5 pencere) → varyansı gör
- [ ] **Nokta metrikleri:** MASE (seasonal, m=24), sMAPE, RMSE, MAE, ND/NRMSE
- [ ] **Olasılıksal metrikler:** WQL (weighted quantile loss), CRPS, coverage@p90 — Chronos & PatchTST-olasılıksal için
- [ ] **Kalibrasyon:** tahmin aralıklarının gerçek kapsama oranı
- [ ] Raporlama: model × metrik tablosu (Markdown/CSV) + tahmin-vs-gerçek grafikleri
- [ ] İstatistiksel anlamlılık: Diebold-Mariano testi (isteğe bağlı)

> Öneri: `utilsforecast`/`datasetsforecast` (Nixtla) metrik fonksiyonlarını kullan;
> tekerleği yeniden icat etme, ama backtest orkestrasyonunu kendin sar.

**DoD:** Sahte tahminle metrik pipeline'ı test edildi; `make eval <run_id>` tablo üretiyor.

---

## Faz 4 — Klasik Baseline'lar  ·  (~2–3 gün)

**Amaç:** Aşılması gereken referans çıtayı koymak. **Basit model şaşırtıcı iyi olabilir.**

- [ ] **Naive** ve **SeasonalNaive** (m=24 ve m=168) — mutlak alt çıta
- [ ] **AutoETS**, **AutoARIMA**, **AutoTheta** — `statsforecast` (321 seri için hızlı, paralel)
- [ ] **Global ML baseline:** LightGBM + lag/rolling/calendar özellikleri — `mlforecast`
- [ ] (Ops.) MSTL — çoklu mevsimsellik (24 + 168) ayrıştırma
- [ ] Her modeli Faz 3 çerçevesinden geçir, MLflow'a logla

**DoD:** Baseline karşılaştırma tablosu; **"yenilmesi gereken skor" = SeasonalNaive/ETS MASE**.

---

## Faz 5 — PatchTST (Transformer)  ·  (~4–5 gün)

**Amaç:** Modern derin öğrenme referansı; foundation modelle deep-learning arası köprü.

- [ ] Uygulama: **`neuralforecast`** (Nixtla) `PatchTST` — hızlı yol; veya orijinal repo
- [ ] Girdi: global model, seri-başına normalizasyon (RevIN/instance norm)
- [ ] Hiperparametreler (deney): patch_len, stride, context_len, d_model, katman, dropout
- [ ] Eğitim: erken durdurma (validation), mixed precision, GPU
- [ ] Olasılıksal çıktı: quantile loss ile p10/p50/p90 (WQL kıyası için)
- [ ] Hiperparametre araması: Optuna (küçük grid ile başla)

**DoD:** PatchTST metrikleri baseline tablosuna eklendi; en iyi konfig `configs/model/patchtst.yaml`.

---

## Faz 6 — Chronos (Time-Series Foundation Model)  ·  (~3–4 gün)

**Amaç:** Foundation modelin bu verideki gücünü ölçmek — asıl merak edilen.

- [ ] **Zero-shot:** `amazon/chronos-bolt-{small,base}` veya `chronos-t5-*` ile eğitimsiz tahmin
  - Bolt: daha hızlı/verimli, önce onu dene
- [ ] Bağlam uzunluğu ve model boyutu duyarlılık analizi
- [ ] Olasılıksal çıktı: doğal olarak kuantil üretir → WQL/CRPS ile değerlendir
- [ ] **Fine-tuning:** bu 321 seride domaine uyarlama; zero-shot vs fine-tuned kıyası
- [ ] Çıkarım maliyeti: gecikme & bellek profili (deployment kararı için kritik)
- [ ] (Ops.) İkinci foundation model ile çapraz kontrol: **TimesFM** veya **Moirai**

**DoD:** Chronos zero-shot + fine-tuned metrikleri tabloda; maliyet/performans notu.

---

## Faz 7 — Karşılaştırma, Hata Analizi ve Model Seçimi  ·  (~2 gün)

**Amaç:** "Kazanan"ı sadece tek metrikle değil, bütünsel seçmek.

- [ ] Ana benchmark tablosu: tüm modeller × (MASE, sMAPE, RMSE, WQL, gecikme, maliyet)
- [ ] Rolling-origin varyans/güven aralıkları; anlamlılık testi
- [ ] Segment bazlı hata analizi: ufuk (h=1…168), günün saati, seri ölçeği/kümesi
- [ ] Nerede hangi model kazanıyor? (ör. kısa ufuk vs uzun ufuk; büyük vs küçük seri)
- [ ] Maliyet–performans dengesi: doğruluk vs gecikme vs altyapı → **prod modeli kararı**
- [ ] Karar kaydı (ADR): hangi model, neden

**DoD:** `reports/benchmark.md` + tercih edilen model gerekçesiyle belirlendi.

---

## Faz 8 — Deployment (Servisleştirme)  ·  (~3–4 gün)

**Amaç:** "Deploy edilebilir" sözünü tutmak — reprodüksiyon değil, ürün.

- [ ] **Model registry:** MLflow ile seçili modelin versiyonlanması
- [ ] **Inference API:** FastAPI — `POST /forecast` (girdi: seri geçmişi + horizon; çıktı: p10/p50/p90)
- [ ] Çıkarım optimizasyonu: batch inference; torch.compile / ONNX / quantization (özellikle Chronos)
- [ ] **Containerization:** Dockerfile + (ops.) docker-compose; sağlık ucu `/health`
- [ ] Yük testi: p50/p95 gecikme, throughput; hedef SLA'ya göre doğrula
- [ ] Reprodüksiyon: `make serve` + örnek `curl` / client
- [ ] (Ops.) Toplu (batch) tahmin işi: tüm seriler için gecelik forecast üret

**DoD:** Container ayağa kalkıyor, örnek istek kalibre tahmin dönüyor, gecikme raporlandı.

---

## Faz 9 — İzleme, MLOps ve Süreklilik  ·  (~2–3 gün, opsiyonel/ileri)

**Amaç:** Ölçülebilir ve yaşayan sistem.

- [ ] **Drift tespiti:** girdi dağılımı ve tahmin hatası kayması izleme
- [ ] Canlı metrik dashboard'u (Grafana/Streamlit): gerçek vs tahmin, rolling MASE
- [ ] Yeniden eğitim tetikleyicisi: performans eşik altına düşünce
- [ ] CI/CD: test + lint + (ops.) model eval gate; `pre-commit` + GitHub Actions
- [ ] Dokümantasyon: model kartı, API dokümanı, çalıştırma rehberi

**DoD:** Basit monitoring paneli + retraining stratejisi yazılı.

---

## Teknoloji Yığını (özet)

| Katman | Araç |
|--------|------|
| Ortam/paket | `uv` + `pyproject.toml`, Python 3.11 |
| Konfig | Hydra / OmegaConf |
| Klasik/ML | `statsforecast`, `mlforecast`, LightGBM |
| Derin öğrenme | `neuralforecast` (PatchTST), PyTorch |
| Foundation | `chronos-forecasting` (Chronos-Bolt), HuggingFace |
| Metrik | `utilsforecast`, kendi backtest sarmalayıcın |
| Takip | MLflow (veya W&B) |
| Servis | FastAPI + Docker; ONNX/torch.compile |
| Kalite | ruff, black, mypy, pytest, pre-commit |

---

## Kritik İlkeler / Tuzaklar

1. **Değerlendirmeyi modelden önce sabitle** (Faz 3). Sonradan metrik/split değişirse tüm kıyas çöker.
2. **Sızıntı yok:** normalizasyon istatistikleri yalnızca train'den; cutoff sonrası veriyi hiçbir aşamada görme.
3. **Seri ölçeği çok değişken** → per-series normalizasyon (RevIN/instance norm) şart.
4. **Baseline'ı ciddiye al:** SeasonalNaive/ETS'yi geçemeyen karmaşık model = başarısızlık.
5. **Olasılıksal değerlendirme** foundation modellerin asıl değeri; sadece nokta metrik bakma.
6. **Zero-shot'ı fine-tuning'den ayrı raporla** — foundation model hikâyesinin özü bu kıyas.
7. **Maliyet bir metriktir:** en doğru model, en pahalı çıkarımsa prod'a uygun olmayabilir.

---

## Önerilen Sıra ve Bağımlılıklar

```
Faz 1 (altyapı) → Faz 2 (veri) → Faz 3 (değerlendirme) ─┐
                                                          ├→ Faz 4 (baseline)
                                                          ├→ Faz 5 (PatchTST)
                                                          └→ Faz 6 (Chronos)
                                                                   ↓
                                              Faz 7 (kıyas & seçim)
                                                                   ↓
                                       Faz 8 (deploy) → Faz 9 (MLOps)
```
Faz 4–5–6 birbirinden bağımsız; paralel ilerletilebilir. Tahmini toplam: **~4–6 hafta**
(tek kişi, yarı zamanlı). Hızlı MVP isteniyorsa: Faz 1→2→3→4→6(zero-shot)→8 minimal hattıyla
1.5–2 haftada uçtan uca demo çıkar, sonra derinleştir.
