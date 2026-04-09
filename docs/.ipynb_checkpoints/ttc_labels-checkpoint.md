# TTC ground-truth labels

This document describes the annotation-derived time-to-collision (TTC) labels produced by `tools/generate_ttc_labels.py`.

## Definition

For each `sample_annotation` on a nuScenes sample:

1. **Horizontal plane only (XY)**
  All geometry uses global-frame XY coordinates. Vertical separation is ignored so TTC is a 2-D “surrogate” contact time in the ground plane.
2. **Range**
  `distance` is the Euclidean distance in the XY plane between the annotated object center and the ego vehicle pose (`ego_pose.translation` on the keyframe linked to `LIDAR_TOP`), matching the ego reference stored in temporal info files as `ego2global_translation`.
3. **Closing speed**
  Let `r_hat` be the unit vector from ego to object in XY, and `v` the object’s horizontal velocity in the **global** frame from `NuScenes.box_velocity` (first two components). The range decrease rate is  
   `closing_speed = -dot(v, r_hat)`.  
   Positive values mean the object is getting closer in projection along the line of sight in XY.
4. **Minimum closing speed (0.5 m/s)**
  Annotations with `closing_speed < 0.5` m/s are **not** labeled. This removes near-stationary noise and non-approaching cases (negative or tiny closing rates) where `distance / closing_speed` is unstable or not meaningful.
5. **Cap (10 s)**
  For valid approaching motion, `TTC = min(distance / closing_speed, 10)` seconds. This bounds influence of outliers and matches the upper range used by the learned head in later phases.

## Output format

The pickle stores:

- `labels`: dict keyed by **annotation token** → `{ttc, distance_xy, closing_speed, sample_token, category}`  
- `metadata`: version, paths, hyperparameters (`ttc_cap_s`, `min_closing_speed_m_s`), and simple coverage stats.

## Heuristic baseline alignment

`tools/ttc_heuristic_baseline.py` maps StreamPETR LiDAR-frame boxes and velocities into the global XY frame with the per-frame `lidar2global` transform derived from the same info fields used to build the dataset, then applies the same TTC definition so predictions are comparable to the stored labels.