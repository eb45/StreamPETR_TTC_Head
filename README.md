# StreamPETR + TTC Risk Head

Frozen StreamPETR detector with a lightweight MLP head that predicts time-to-collision (TTC) in seconds on nuScenes.

## What it Does

<p align="center">
  <img src="docs/imgs/mermaid_diagram_372.svg" alt="Flowchart: Part 1 ground-truth TTC labels, Part 2 physics TTC from StreamPETR predictions, Part 3 frozen StreamPETR with TTC MLP head and supervision from GT TTC" width="92%" />
</p>

This project uses time-to-collision (TTC) as a risk metric for autonomous driving and evaluates it on the [nuScenes](https://www.nuscenes.org/) dataset with StreamPETR as the 3D perception backbone. Ground-truth TTC is built from physics-based labels. We then compare two predictors: a physics baseline that feeds StreamPETR’s outputs through the same closure model used for labeling, and a TTC head trained on top of a frozen StreamPETR so the network regresses seconds-to-collision directly from object query features—giving you both an interpretable baseline and a learned risk estimate from the same detections.

<p align="center">
  <img src="docs/imgs/scene_ttc_front_example.gif" alt="CAM_FRONT qualitative demo: TTC-colored boxes from compare_ttc_scene" width="92%" />
</p>

## Quick Start

1. **[SETUP.md](SETUP.md)** — environment, nuScenes layout, temporal infos, TTC label pickle, and StreamPETR weights in `ckpts/`.
2. **Train (Slurm):** `sbatch ttc_mlp_head.sh` (1 GPU) or `sbatch ttc_mlp_head_4gpu.sh` — set `NUSCENES_ROOT` and related env vars as documented there.
3. **Eval (Slurm):** `CHECKPOINT=work_dirs/.../latest.pth sbatch run_eval_ttc_mlp.sh` for mini val; `sbatch run_eval_ttc_mlp_full.sh` for full val (optional `NUSCENES_ROOT`).

## Video Links

- **Demo:** *[ADD DEMO VIDEO URL HERE]*
- **Technical walkthrough:** *[ADD TECHNICAL WALKTHROUGH HERE]*

## Data

Used [nuScenes](https://www.nuscenes.org/) autonomous driving dataset. It consists of about **1,000 scenes of 20 s each**, on the order of **1,400,000** camera images and **390,000** LiDAR sweeps, recorded in **Boston and Singapore** with both **left- and right-hand** traffic. In this repo,  **v1.0-mini** is used for quick tests and **v1.0-trainval** for full experiments.

<p align="center">
  <img src="docs/imgs/nuscenes_dataset_example.png" alt="Example scenes from Nuscenes dataset" width="92%" />
</p>


| Split | Pkl filename                          | Keyframes           |
| ----- | ------------------------------------- | ------------------- |
| train | `nuscenes2d_temporal_infos_train.pkl` | 28,130 (700 scenes) |
| val   | `nuscenes2d_temporal_infos_val.pkl`   | 6,019 (150 scenes)  |
| test  | `nuscenes2d_temporal_infos_test.pkl`  | 6,008 (150 scenes)  |

**TTC supervision (labels, `LoadGTTC`, matching, loss):** see **[docs/ttc_supervision_pipeline.md](docs/ttc_supervision_pipeline.md)**.


## Evaluation

## Protocol

All quantitative results use the **1000-batch / 2779-pair** protocol on the full `v1.0-trainval` val split (`data_full`). Two evaluation scripts are used and reported separately:

- `eval_ttc_breakdown` — per-pair MAE and RMSE between predicted TTC and GT TTC, matched by annotation token
- `eval_ttc_mlp` — mean sum of `loss_ttc` terms per batch (includes the loss weighting scheme)

The **physics baseline** applies the same closure model used to generate labels (`distance / closing_speed`, capped at 10 s) to StreamPETR's predicted boxes and velocities . All comparisons are against GT TTC derived from `generate_ttc_labels.py`.

---

## 1. Primary Ablation Study

> Key questions:
>
> 1. Does the learned head improve over the physics heuristic, and by how much?
> 2. TTC Head design: Does adding velocity to the head improve performance?


| Predictor                                    | n_pairs | MAE (s) | RMSE (s) | Mean error (s) |
| -------------------------------------------- | ------- | ------- | -------- | -------------- |
| Physics baseline                             | 2779    | —       | —        | —              |
| MLP head (Only using query embeddings)       | 2779    | —       | —        | —              |
| MLP head (Using query + velocity embeddings) | 2779    | —       | —        | —              |


---

## 2. Conditional breakdown by GT TTC bin

> Key question: does the MLP head outperform the physics baseline in the safety-critical short bins ([0, 1) and [1, 3))?

Results split by GT TTC bin, aligned with the loss tiers used during training. The [0, 1) bin has very few samples and its numbers should be interpreted with caution.


| GT TTC bin (s)   | n   | Physics MAE / RMSE | MLP MAE / RMSE |
| ---------------- | --- | ------------------ | -------------- |
| [0, 1)           | —   | —                  | —              |
| [1, 3)           | —   | —                  | —              |
| [3, 10)          | —   | —                  | —              |
| [10, ∞) (capped) | —   | —                  | —              |


---

## 3. Per-class breakdown

MAE per object class for both predictors. Pedestrians and cyclists involve more complex motion that the simple closing-speed model may struggle with; cones and barriers are nearly static and should be near the 10 s cap.


| Class      | n   | Physics MAE (s) | MLP MAE (s) |
| ---------- | --- | --------------- | ----------- |
| car        | —   | —               | —           |
| pedestrian | —   | —               | —           |
| bicycle    | —   | —               | —           |
| motorcycle | —   | —               | —           |
| truck      | —   | —               | —           |
| bus        | —   | —               | —           |
| trailer    | —   | —               | —           |


---

## 4. Qualitative evaluation

Four scenes chosen to stress-test different driving regimes. Each panel shows CAM_FRONT with TTC-colored bounding boxes annotated with three values per object: GT / physics / MLP. BEV panels shown where available.


| Scene         | Regime                                        | What to look for                                                       |
| ------------- | --------------------------------------------- | ---------------------------------------------------------------------- |
| Intersection  | Fast-closing cross-traffic, direction changes | Does the head handle non-radial approach vectors?                      |
| Straight-away | Same-direction traffic, slow closing speed    | Are near-static objects correctly filtered or capped at 10 s?          |
| Parking lot   | Near-static objects, low closing speeds       | Checks label filtering and cap behavior                                |
| Near-miss     | High-urgency short-TTC event                  | Does the head flag risk earlier or more accurately than the heuristic? |


> Scene comparisons generated with `compare_ttc_scene`; see `notebooks/` for links to BEV/CAM panel outputs.

---

Based on [StreamPETR](https://github.com/exiawsh/StreamPETR) (ICCV 2023).

```bibtex
@article{wang2023exploring,
  title={Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection},
  author={Wang, Shihao and Liu, Yingfei and Wang, Tiancai and Li, Ying and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2303.11926},
  year={2023}
}
```

