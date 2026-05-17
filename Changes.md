# Changes: Knowledge Distillation (KD)

Knowledge distillation was added on top of the existing CenterPoint training stack. Two variants exist; only one is used per config:

- **Response-based KD** (`kd.type = "heatmap_mse"`): MSE between student and teacher **heatmap** outputs, weighted by `lambda_kd`.
- **Feature-based KD** (`kd.type = "feature_mse"`): MSE on **`CenterHead.shared_conv`** features, weighted by `lambda_feat`.

The original detection loss (focal heatmap + L1 regression from `centernet_loss.py`) is unchanged. KD terms are added in `CenterHead.loss`. At test time only the student checkpoint is used.

---

## Code changes

| File | Change |
|------|--------|
| `det3d/torchie/apis/train.py` | If `cfg.kd.enabled`, build teacher from `teacher_config`, load `teacher_checkpoint`, freeze it, pass `teacher_model` and `kd_cfg` to the trainer. |
| `det3d/torchie/trainer/trainer.py` | Each training step: teacher forward under `no_grad` (`return_preds` and/or `return_feats`), then student forward with teacher outputs and `kd_cfg`. |
| `det3d/models/detectors/point_pillars.py` | Return `head_shared` from the head; support `return_preds` / `return_feats` for the teacher; pass KD arguments into `bbox_head.loss`. |
| `det3d/models/detectors/voxelnet.py` | Same KD-related forward/loss wiring as PointPillars. |
| `det3d/models/bbox_heads/center_head.py` | In `loss()`: compute `hm_kd_loss` (heatmap MSE) or `feat_kd_loss` (shared-feature MSE) and add to the per-task loss; log both metrics. |
| `det3d/torchie/trainer/hooks/logger/text.py` | Log `hm_kd_loss` and `feat_kd_loss` with 6 decimal places. |
| `det3d/torchie/apis/env.py` | Device selection limited to CUDA or CPU (MPS removed). |

**Unchanged:** `det3d/models/losses/centernet_loss.py` (baseline `FastFocalLoss`, `RegLoss`).

---

## New configs (`configs/nusc/pp/`)

Each file sets a slimmer student (reduced reader/neck/head channels) and a `kd` block (`enabled`, `type`, `lambda_kd` or `lambda_feat`, `teacher_config`, `teacher_checkpoint`).

**Response-based:** `response_based_kd.py`, `response_based_kd_05.py`, `response_based_kd_08.py`, `response_based_kd_smoke.py`, `response_based_kd_resnet.py`, `response_based_kd_resnet_smoke.py`

**Feature-based:** `feature_based_kd.py`, `feature_based_kd_05.py`, `feature_based_kd_08.py`, `feature_based_kd_smoke.py`
