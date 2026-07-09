# Model Karşılaştırma Raporu

Saatlik elektrik talebi tahmini benchmark'ının sonuç analizi. Tüm sayılar donuk değerlendirme
protokolünden (5 rolling-origin haftalık pencere × 168 saatlik ufuk, 321 seri) gelir; ham tablolar
`outputs/metrics/` altındadır ve her bölümde kaynağı belirtilmiştir. Koşu: Colab A100, 2026-07-09
(`run_meta.json`).

## 1. Genel sıralama (`metrics/leaderboard.csv`)

| model | MASE | sMAPE | WQL | coverage@80 | bar'a göre |
|---|---|---|---|---|---|
| **chronos_bolt_base_ft** | **0.9358** | 10.83 | **0.0896** | 0.701 | **−15.9%** |
| chronos_bolt_base | 0.9378 | 10.68 | 0.0900 | 0.712 | −15.7% |
| chronos_bolt_small | 0.9754 | 11.11 | 0.0920 | 0.710 | −12.4% |
| seasonal_naive_168 (bar) | 1.1130 | 11.71 | 0.1279 | 0.095 | 0 |
| lgbm_global | 1.2073 | 14.40 | 0.2828 | 0.974 | +8.5% |
| mstl | 1.2984 | 17.01 | 0.1398 | 0.884 | +16.7% |
| seasonal_naive_24 | 1.2993 | 14.33 | 0.1527 | 0.045 | +16.7% |
| auto_ets | 1.3988 | 17.40 | 0.2335 | 0.808 | +25.7% |
| patchtst | 1.7116 | 20.10 | 0.1517 | 0.721 | +53.8% |
| auto_theta | 1.7528 | 19.93 | 0.2337 | 0.956 | +57.5% |

**Ana bulgu:** Projenin sorusu netçe cevaplandı — *yalnızca* foundation model ailesi "geçen haftayı
tekrarla" barajını geçebildi. Üç Chronos varyantı MASE < 1 (örneklem-içi seasonal naive ölçeğinden
bile iyi); hiçbir klasik model, global LightGBM ya da sıfırdan eğitilen PatchTST barajı geçemedi.
Olasılıksal kalitede fark daha da büyük: Chronos'un WQL'i (~0.090), kalibrasyonu sağlıklı en iyi
klasik alternatifin (MSTL 0.140) %36 altında.

## 2. Pencere kırılımı — tatil etkisi (`metrics/*_by_window.csv`)

MASE, pencere kesim tarihine göre:

| cutoff | snaive168 | lgbm | patchtst | chronos base | chronos ft |
|---|---|---|---|---|---|
| 26 Kas | 0.897 | 0.910 | 1.154 | 0.716 | 0.728 |
| 03 Ara | 1.009 | 1.051 | 1.232 | 0.812 | 0.812 |
| 10 Ara | 1.014 | 1.120 | 1.181 | 0.821 | 0.831 |
| 17 Ara | 1.080 | 1.197 | 1.517 | 1.024 | 1.026 |
| **24 Ara (Noel)** | **1.565** | **1.759** | **3.474** | **1.317** | **1.282** |

- Noel haftası herkes için en zor pencere; ama modeller arasındaki ayrışma burada uçuruma dönüşüyor:
  PatchTST 3.47'ye çökerken (takvim bilgisi yok, tatil deseni öğrenilemedi) Chronos 1.28–1.32 ile
  bozulmayı sınırlıyor.
- Chronos her beş pencerede de sıralamanın tepesinde — üstünlük tek bir "şanslı" haftadan gelmiyor.

## 3. Fine-tuning'in katkısı (ft vs zero-shot base)

- Toplam fark marjinal: MASE 0.9358 vs 0.9378 (‰2), WQL 0.0896 vs 0.0900. Fine-tune erken durdu
  (pencere başına 350–650 adım, final 400; `train_logs/ft_*_val_curve.csv`) — ön-eğitim bu veri
  desenini zaten büyük ölçüde kapsıyor.
- **Kazanç nerede:** pencere bazında delta (ft − base): +0.012, −0.001, +0.010, +0.002, **−0.034**.
  Yani ft, dört "normal" haftada base ile başa baş ya da kıl payı geride; tek anlamlı kazancı
  **en zor pencere olan Noel haftasında** (1.317 → 1.282). Operasyonel açıdan bu değerli bir
  özellik: tahminin en kritik olduğu anomalili dönemde daha dayanıklı.
- Seri×pencere kafa kafaya: ft, base'i çiftlerin yalnızca %44'ünde geçiyor (WQL'de %46) — toplamdaki
  birincilik geniş tabanlı değil, zor pencerede yoğunlaşmış. Base'e karşı "her yerde daha iyi"
  iddiası kurulamaz; doğru iddia "aynı ortalama, zor haftada daha sağlam"dır.
- **Zor haftadaki kazanç istatistiksel olarak anlamlı ve geniş tabanlı:** Noel penceresinde ft,
  321 serinin %58'inde (187/321) base'den iyi (işaret testi p≈0.002; WQL'de %57); iyileşmelerin
  çeyreği seri başına 0.08 MASE'den büyük. Normal pencerelerde bu oran %33–46'ya düşüyor. Mekanizma
  tutarlı: fine-tune verisi 2012–2013 Noel haftalarını içeriyordu — model bu şebekenin kendi tatil
  davranışını öğrendi. Dikkat: test dönemindeki tek istisnai bölüm bu (tek tatil epizodu); "tüm
  istisnai dönemlerde daha iyi" genellemesi tek epizoda dayanır, ihtiyatla kurulmalıdır.

## 4. Kazanmanın genişliği ve ölçek duyarlılığı (`metrics/*_per_series.csv`)

- ft, seasonal naive'i seri×pencere çiftlerinin **%78'inde** geçiyor (base %79) — üstünlük birkaç
  büyük seriden değil, dağılımın genelinden geliyor. PatchTST'ye karşı kazanma oranı %95.
- Ölçek gruplarına göre (seri ortalamasına göre üçe bölünmüş): ft MASE küçük/orta/büyük serilerde
  0.924 / 0.949 / 0.934 — ölçekten bağımsız, dengeli. (Karşılaştırma: snaive 1.107 / 1.138 / 1.094.)

## 5. Kalibrasyon (coverage@80, nominal 0.80)

| model | coverage@80 | okuma |
|---|---|---|
| chronos ailesi | 0.70–0.71 | aralıklar ~9 puan dar → alt/üst kantiller hafif iyimser |
| auto_ets | 0.808 | neredeyse mükemmel kalibrasyon (ama nokta doğruluğu zayıf) |
| mstl | 0.884 | hafif geniş |
| lgbm / auto_theta | 0.97 / 0.96 | aşırı geniş aralıklar (WQL'i de bu bozuyor) |
| seasonal naive'ler | 0.05–0.10 | dejenere (nokta model, aralık üretmiyor) |

**Öneri:** Chronos üretimde kullanılacaksa ve kapsama garantisi gerekiyorsa, aralıklar conformal
bir düzeltmeyle (geçmiş artıklardan kantil genişletme) kalibre edilmelidir; model kartında da bu
uyarı yer alıyor.

## 6. PatchTST'nin hikâyesi (dürüst dipnot)

İlk CPU koşusu (500 adım tavanı, early stopping hiç tetiklenmedi) MASE 1.97 vermişti; adil GPU
bütçesiyle (early stopping 1.6k–2.5k adımda durdu) 1.71'e geldi ama barajı yine geçemedi. Kırılım,
nedeni gösteriyor: dört normal haftada 1.15–1.52 (bar'a yakın), Noel'de 3.47 (çöküş). Eğitim
bütçesi gerçek bir sorundu ve düzeltildi; kalan açık, mimarinin takvim/tatil bilgisine kör
olmasıdır. Bu, "kısıtlı veriyle sıfırdan eğitilen transformer, güçlü mevsimsel naive'i geçemez"
literatür bulgusuyla uyumludur.

## 7. Sonuç ve model seçimi

1. **Servis edilecek model:** `chronos_bolt_base_ft` (`outputs/models/chronos_bolt_base_ft/`,
   bf16 safetensors + model kartı). Gerekçe: leaderboard birincisi, en zor pencerede en dayanıklı,
   ölçekten bağımsız dengeli. Zero-shot base makul bir yedek/karşılaştırma noktasıdır (fark ‰2).
2. **Basit ve ucuz alternatif:** hiçbiri kurulamıyorsa `seasonal_naive_168` hâlâ saygın bir
   temeldir; tüm klasik/ML modeller ondan kötüdür.
3. **Açık iş:** kantil kalibrasyonu (conformal) ve istenirse tatil-takvimli bir varyant
   (exogenous destekli model ya da tatil-bilinçli sonradan düzeltme).
