from pathlib import Path

# Uses the base config and overrides for student architecture and feature-based KD
_base_cfg = Path(__file__).with_name("baseline.py")
exec(_base_cfg.read_text(), globals(), globals())

# Smaller student architecture
model["reader"]["num_filters"] = [32, 32]
model["backbone"]["num_input_features"] = 32
model["neck"]["num_input_features"] = 32
model["neck"]["ds_num_filters"] = [32, 64, 128]
model["neck"]["us_num_filters"] = [64, 64, 64]
model["bbox_head"]["in_channels"] = sum([64, 64, 64])

# Feature KD (CenterHead shared_conv output MSE vs teacher)
kd = dict(
    enabled=True,
    type="feature_mse",
    lambda_feat=0.8,
    teacher_config="./configs/nusc/pp/baseline.py",
    # teacher_checkpoint="./work_dirs/baseline_smoke/latest.pth",
    teacher_checkpoint="../Computer-Vision/work_dirs/nusc_centerpoint_pp_02voxel_two_pfn_10sweep/latest.pth",
)

work_dir = "./work_dirs/feature_based_kd/"
