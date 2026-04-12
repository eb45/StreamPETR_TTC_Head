# StreamPETR + TTC Risk Head

Frozen StreamPETR backbone with a small MLP for time-to-collision (seconds) on nuScenes.

**Docs:** [setup](docs/setup.md) · [data](docs/data_preparation.md) · [TTC labels](docs/ttc_labels.md)

Use one **nuScenes split** end-to-end: **mini** (`v1.0-mini`, smaller, faster) or **full trainval** (`v1.0` / `v1.0-trainval` trees, much longer runs). Match `--version` flags and pickle paths to the data you actually downloaded.

| Step | Stage | What to run |
|------|--------|-------------|
| 1 | Environment | Follow [docs/setup.md](docs/setup.md); put nuScenes under `data/nuscenes/`. |
| 2 | Temporal infos | `python tools/create_data_nusc.py --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes2d --version <ver>` — **mini:** `v1.0-mini` · **full:** `v1.0` (see [data prep](docs/data_preparation.md)). Writes `nuscenes2d_temporal_infos_{train,val}.pkl`. |
| 3 | Pretrained detector | Place StreamPETR weights in `ckpts/` (e.g. `stream_petr_vov_flash_800_bs2_seq_24e.pth`). |
| 4 | TTC labels | `python tools/generate_ttc_labels.py --data-root ./data/nuscenes --version <ver> --out <path.pkl>` — **mini:** e.g. `v1.0-mini` → `ttc_gt_labels_v1_0_mini.pkl` · **full:** e.g. `v1.0-trainval` → e.g. `ttc_gt_labels_v1_0_trainval.pkl`. |
| 5 | Train TTC head | `python tools/train.py projects/configs/StreamPETR/stream_petr_vov_ttc_frozen_20e.py --work-dir ./work_dirs/... --launcher none` **or** Slurm: `sbatch ttc_mlp_head.sh` (1 GPU) / `sbatch ttc_mlp_head_4gpu.sh` (4 GPU). Set `NUSCENES_VER` in Slurm to match your split. |
| 6 | Baselines | `python tools/ttc_heuristic_baseline.py` (`--help` for args). |
| 7 | Eval TTC | `python tools/eval_ttc_breakdown.py ... <work_dir>/latest.pth --ann-file ./data/nuscenes/nuscenes2d_temporal_infos_val.pkl` — same release as training; mini and full each have their own `*_val.pkl`. |

Set **`STREAMPETR_TTC_PKL`** (step 4 pickle) and **`STREAMPETR_LOAD_FROM`** (step 3 checkpoint) when training or evaluating.

Based on [StreamPETR](https://github.com/exiawsh/StreamPETR) (ICCV 2023).

```bibtex
@article{wang2023exploring,
  title={Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection},
  author={Wang, Shihao and Liu, Yingfei and Wang, Tiancai and Li, Ying and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2303.11926},
  year={2023}
}
```
