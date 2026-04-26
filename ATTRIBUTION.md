# Attribution

## Upstream projects and code

- **StreamPETR** — Multi-view 3D detection with temporal modeling; base architecture and training patterns.
  - Repository: [https://github.com/exiawsh/StreamPETR](https://github.com/exiawsh/StreamPETR)  
  - Paper: Wang et al., *Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection*, arXiv:2303.11926, 2023.
- **OpenMMLab MMDetection3D** — Data pipelines, nuScenes adapters, and detection framework.  
  - [https://github.com/open-mmlab/mmdetection3d](https://github.com/open-mmlab/mmdetection3d)

## Datasets

- **nuScenes** — Autonomous driving dataset (cameras, lidar, maps, annotations).  
  - Caesar et al., *nuScenes: A multimodal dataset for autonomous driving*, CVPR 2020.  
  - [https://www.nuscenes.org/](https://www.nuscenes.org/)

## Third-party libraries:

Core stack  includes:

- **PyTorch**, **torchvision**  
- **mmcv-full**, **mmdet**, **mmsegmentation**, **mmdet3d**  
- **NumPy**, **SciPy** (and common scientific Python deps pulled in by MMDet3D)

Optional:

- **flash-attn** (optional acceleration; hardware-dependent)  
- **Weights & Biases** (optional experiment logging)

## AI-generated material

- Slurm job scripts, shell wrappers, etc.
- Adding docstrings to files
- Structure of the notebooks
- Debugging help (path resolutions, fixing OOM errors, etc.)
- Repo maintenance like generating requirements.txt file
- Some of the more mathematical portions like the equations for generating TTC labels