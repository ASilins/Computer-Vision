from pathlib import Path

# Keep smoke runs aligned with the real training setup by loading
# the main PointPillars config first, then overriding only runtime knobs.
_base_cfg = Path(__file__).with_name("baseline.py")
exec(_base_cfg.read_text(), globals(), globals())

# Run a tiny local loop quickly while preserving model/data pipeline structure.
total_epochs = 1

data["samples_per_gpu"] = 1
data["workers_per_gpu"] = 0

# Subsample infos to reduce the number of train/val samples for smoke tests.
for split in ("train", "val", "test"):
    if split in data and isinstance(data[split], dict):
        data[split]["load_interval"] = 20

log_config["interval"] = 1
checkpoint_config["interval"] = 1
work_dir = "./work_dirs/baseline_smoke/"
