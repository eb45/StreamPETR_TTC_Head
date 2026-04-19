# Installation and setup

This project builds on **PyTorch**, **MMDetection3D**, and **StreamPETR**. The dataset used is **nuScenes**. You can optionally use **Weights & Biases** for logging.

## 1. System requirements

- **Python** ≥ 3.8  
- **CUDA** compatible with your PyTorch build  
- **GPU** strongly recommended for training and evaluation (large image volume on full nuScenes)

## 2. Conda environment (recommended)

```bash
conda create -n streampetr python=3.8 -y
conda activate streampetr
```

Install PyTorch and torchvision for your CUDA version:

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Optional: **flash-attn** can speed up attention and save memory.

## 3. MMDetection / MMDetection3D

Follow [MMDetection3D installation](https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation) steps:

```bash
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
```

```bash
cd mmdetection3d
pip install -e .
```

## 4. Python dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

## 5. Data: nuScenes

1. Download **nuScenes** from [https://www.nuscenes.org/download](https://www.nuscenes.org/download) (mini for quick tests; trainval for full experiments).
2. Extract so scenes live under your chosen root (e.g. `data/nuscenes/` or a shared HPC path).
3. Generate temporal infos (mini vs full differs by `--version`):

```bash
python tools/create_data_nusc.py \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes2d \
  --version v1.0-mini
```

See [docs/data_preparation.md](docs/data_preparation.md) for `v1.0` / trainval layouts and optional prebuilt `.pkl` downloads.

## 6. Pretrained detector weights

Create `ckpts/` at the project root and place a StreamPETR checkpoint (for example `stream_petr_vov_flash_800_bs2_seq_24e.pth`). Point training/evaluation to it with `STREAMPETR_LOAD_FROM` (see the Slurm scripts’ comments and your config).

## 7. TTC labels

Generate labels with:

```bash
python tools/generate_ttc_labels.py \
  --data-root ./data/nuscenes \
  --version v1.0-mini \
  --out ./data/nuscenes/ttc_gt_labels_v1_0_mini.pkl
```

Set `STREAMPETR_TTC_PKL` to that path. Details: [docs/ttc_labels.md](docs/ttc_labels.md).

## 8. Training (Slurm)

Run jobs from the **repository root**

**1 GPU**

```bash
cd /path/to/StreamPETR

export NUSCENES_ROOT="/path/to/StreamPETR/data_full/nuscenes/"
export NUSCENES_ROOT="${NUSCENES_ROOT%/}/"

sbatch ttc_mlp_head.sh
```

**4 GPU**

```bash
cd /path/to/StreamPETR

export NUSCENES_ROOT="/path/to/StreamPETR/data_full/nuscenes/"
export NUSCENES_ROOT="${NUSCENES_ROOT%/}/"

sbatch ttc_mlp_head_4gpu.sh
```

## 9. Evaluation (Slurm)

**Mini val split** (same infos + TTC pickle as nuScenes mini prep) — pass a checkpoint relative to the repo root:

```bash
CHECKPOINT=work_dirs/streampetr_ttc_frozen_20e_4gpu/latest.pth MAX_BATCHES=2 BREAKDOWN_MAX_BATCHES=2 \
sbatch slurm_scripts/run_eval_ttc_mlp.sh
```

**Full trainval val split**:

```bash
CHECKPOINT=work_dirs/streampetr_ttc_frozen_20e_4gpu/latest.pth MAX_BATCHES=2 BREAKDOWN_MAX_BATCHES=2 \
sbatch slurm_scripts/run_eval_ttc_mlp_full.sh
```


More information can be found in the original StreamPETR documentation as well: [docs/setup.md](docs/setup.md), [docs/data_preparation.md](docs/data_preparation.md), [docs/training_inference.md](docs/training_inference.md).