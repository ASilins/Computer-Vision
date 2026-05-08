import logging
import os
import random
import subprocess

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from det3d.torchie.trainer import get_dist_info


def get_train_device(local_rank=0):
    """Pick training device: CUDA (per rank) if available, else CPU."""
    forced = os.environ.get("CENTERPOINT_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        if forced == "cuda" and torch.cuda.is_available():
            return torch.device("cuda", int(local_rank))
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda", int(local_rank))
    return torch.device("cpu")


def init_dist(launcher, backend="nccl", **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    if launcher == "pytorch":
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == "mpi":
        _init_dist_mpi(backend, **kwargs)
    elif launcher == "slurm":
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError("Invalid launcher type: {}".format(launcher))


def _init_dist_pytorch(backend, **kwargs):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    raise NotImplementedError


def _init_dist_slurm(backend, port=29500, **kwargs):
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        "scontrol show hostname {} | head -n1".format(node_list)
    )
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["RANK"] = str(proc_id)
    dist.init_process_group(backend=backend)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=log_level
        )
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel("ERROR")
    return logger
