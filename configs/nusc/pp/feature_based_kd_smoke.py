from pathlib import Path

_base_cfg = Path(__file__).with_name("feature_based_kd.py")
exec(_base_cfg.read_text(), globals(), globals())

total_epochs = 1

data["samples_per_gpu"] = 1
data["workers_per_gpu"] = 0

for split in ("train", "val", "test"):
    if split in data and isinstance(data[split], dict):
        data[split]["load_interval"] = 20

log_config["interval"] = 1
checkpoint_config["interval"] = 1
work_dir = "./work_dirs/feature_based_kd_smoke/"
