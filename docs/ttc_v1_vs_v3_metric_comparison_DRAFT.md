# v1 vs v3 — metric comparison (from your JSONs)

**v1 column** uses: `ttc_loss_metrics (2).json`, `ttc_breakdown (3).json`  
**v3 column** uses: `ttc_loss_metrics v3.json`, `v3_ttc_breakdown.json` (unchanged from before)

> **Protocol:** v1 **breakdown** and v1 **loss** are both **1000 batches** on `data_full` val (`max_batches: 1000`). v3 **breakdown** is **1000 batches / 2779 pairs** (same pair count as v1 breakdown). v3 **loss** JSON is still **100** batches — re-run v3 `eval_ttc_mlp` with `max_batches: 1000` for a strict match to v1 loss averaging.

| | v1 (new files) | v3 |
|---|----------------|-----|
| Breakdown `n_batches` | 1000 | 1000 |
| Breakdown `n_pairs` | 2779 | 2779 |
| Loss `n_batches` | **1000** | **100** |

---

## 1. `eval_ttc_breakdown` (MAE / RMSE, MLP vs GT)

| Metric | **v1** | **v3** |
|--------|--------|--------|
| **n_pairs** | 2779 | 2779 |
| **MAE (s)** | **3.660** | 3.788 |
| **RMSE (s)** | **4.076** | 4.251 |

**By GT TTC bin (seconds)**

| Bin | v1: n / MAE / RMSE | v3: n / MAE / RMSE |
|-----|--------------------|--------------------|
| [0,1) | 4 / 4.10 / 4.10 | 4 / 3.86 / 3.87 |
| [1,3) | 278 / 2.75 / 2.81 | 278 / 2.51 / 2.57 |
| [3,10) | 941 / **1.60** / 2.05 | 942 / 1.65 / 2.16 |
| [10,∞) | 1556 / 5.07 / 5.07 | 1555 / 5.31 / 5.32 |

**By class (MAE s)**

| Class | v1 (n) | v3 (n) |
|-------|--------|--------|
| car | 2.73 (1420) | 2.76 (1419) |
| truck | 3.26 (154) | 3.37 (154) |
| bus | 3.76 (76) | 3.91 (76) |
| trailer | 3.75 (17) | 3.93 (17) |
| motorcycle | 3.76 (29) | 3.97 (29) |
| bicycle | 2.09 (19) | 2.26 (19) |
| pedestrian | 4.98 (1064) | 5.22 (1065) |

---

## 2. `eval_ttc_mlp` (mean sum of `loss_ttc` terms per batch)

| Metric | **v1** | **v3** |
|--------|--------|--------|
| `n_batches` | 1000 | 100 |
| **trained** mean ∑`loss_ttc` / batch | **2.766** | 2.793 |
| **baseline** mean (pretrained) | 3.227 | 3.330 |

*v3 loss averaged over fewer batches — treat as weak comparison until v3 is re-run with 1000 batches.*

---

## 3. Checkpoints & configs (from JSONs)

- **v1 / `ttc_loss_metrics (2).json`:**  
  `.../work_dirs/streampetr_ttc_v2_frozen_20e_1gpu/latest.pth`  
  (path name contains **v2_1gpu**; config in file: `.../stream_petr_vov_ttc_frozen_20e.py`.)

- **v3 / `ttc_loss_metrics v3.json`:**  
  `.../work_dirs/streampetr_ttc_v3_frozen_20e_4gpu/latest.pth`

---

**Source paths used for this version**

- v1: `/Users/emmab/Downloads/ttc_loss_metrics (2).json`, `ttc_breakdown (3).json`
- v3: `ttc_loss_metrics v3.json`, `v3_ttc_breakdown.json` (Downloads, as in your earlier message)
