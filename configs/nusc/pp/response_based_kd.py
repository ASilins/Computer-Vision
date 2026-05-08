from pathlib import Path

# Uses the base config and overrides for student architecture and response-based KD parent
_base_cfg = Path(__file__).with_name("baseline.py")
exec(_base_cfg.read_text(), globals(), globals())

# Smaller student architecture
model["reader"]["num_filters"] = [32, 32]
model["backbone"]["num_input_features"] = 32
model["neck"]["num_input_features"] = 32
model["neck"]["ds_num_filters"] = [32, 64, 128]
model["neck"]["us_num_filters"] = [64, 64, 64]
model["bbox_head"]["in_channels"] = sum([64, 64, 64])

# Heatmap KD parent settings
kd = dict(
    enabled=True,
    type="heatmap_mse",
    lambda_kd=0.2,
    teacher_config="./configs/nusc/pp/baseline.py",
    # teacher_checkpoint="./work_dirs/baseline_smoke/latest.pth",
    teacher_checkpoint="../Computer-Vision/work_dirs/nusc_centerpoint_pp_02voxel_two_pfn_10sweep/latest.pth",
)

work_dir = "./work_dirs/response_based_kd/"
