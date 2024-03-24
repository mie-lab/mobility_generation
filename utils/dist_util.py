"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf

import torch
import torch.distributed as dist

# Change this to reflect your cluster layout.


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    backend = "gloo"
    hostname = "localhost"

    if os.environ.get("LOCAL_RANK") is None:
        os.environ["MASTER_ADDR"] = hostname
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        port = _find_free_port()
        os.environ["MASTER_PORT"] = str(port)
        os.environ["LOCAL_RANK"] = str(0)

    dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():  # This clears remaining caches in GPU 0
        torch.cuda.set_device(get_device())
        torch.cuda.empty_cache()


def get_device():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    return torch.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    # if int(os.environ['LOCAL_RANK']) == 0:
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return torch.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
