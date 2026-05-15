from pathlib import Path

# Response-based KD: teacher = PointPillars (PFN) + ResNetNeck (matches cluster
# nusc_centerpoint_pp_02voxel_two_pfn_10sweep_resnet). Student = narrower PFN + ResNetNeck.
_base_cfg = Path(__file__).with_name("resnet.py")
exec(_base_cfg.read_text(), globals(), globals())

model["reader"]["num_filters"] = [32, 32]
model["backbone"]["num_input_features"] = 32
model["neck"]["num_input_features"] = 32
model["neck"]["ds_num_filters"] = [32, 64, 128]
model["neck"]["us_num_filters"] = [64, 64, 64]
model["bbox_head"]["in_channels"] = sum([64, 64, 64])

kd = dict(
    enabled=True,
    type="heatmap_mse",
    lambda_kd=0.2,
    teacher_config="./configs/nusc/pp/resnet.py",
    teacher_checkpoint="../Computer-Vision/work_dirs/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_resnet/latest.pth",
)

work_dir = "./work_dirs/response_based_kd_resnet/"
