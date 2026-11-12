# src/pipeline/checkpoint.py
"""Resume checkpoints: full training state, written atomically each epoch."""

import dataclasses
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

RESUME_FILENAME = "resume_checkpoint.pt"

# Training fields that may change between sessions without invalidating a resume.
_HASH_EXCLUDE = {"patience"}


def resume_path(config) -> Path:
    return Path(config.paths.model_dir) / RESUME_FILENAME


def training_config_hash(config) -> str:
    """Hash of training settings that determine tasks, LR schedule, and loop behavior."""
    tc = config.training
    d = dataclasses.asdict(tc) if dataclasses.is_dataclass(tc) else dict(vars(tc))
    d = {k: v for k, v in d.items() if k not in _HASH_EXCLUDE}
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _get_rng_states() -> dict:
    s = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_states(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_resume_checkpoint(config, model, trainer, scheduler, epoch: int,
                           loop_state: dict, history: dict, timing: dict) -> None:
    """Write full training state after `epoch` has completed. Atomic."""
    path = resume_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,                       # last *completed* epoch
        "config_hash": training_config_hash(config),
        "model_state": model.model.state_dict(),
        "optimizer_state": trainer.opt.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "loop_state": loop_state,
        "history": history,
        "timing": timing,
        "rng_states": _get_rng_states(),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)                     # atomic on POSIX and Windows


def load_resume_checkpoint(config, model, trainer, scheduler):
    """
    Restore state in place. Returns the checkpoint dict (epoch, loop_state,
    history, timing), or None if no resume checkpoint exists.
    """
    path = resume_path(config)
    if not path.exists():
        return None

    # weights_only=False: the payload holds numpy RNG state and plain Python
    # containers. Safe because we wrote this file ourselves.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    saved, current = ckpt["config_hash"], training_config_hash(config)
    if saved != current:
        raise ValueError(
            f"Training config changed since checkpoint (saved={saved}, "
            f"current={current}). Refusing to resume. Revert the config "
            f"or start a new run."
        )

    model.model.load_state_dict(ckpt["model_state"])
    trainer.opt.load_state_dict(ckpt["optimizer_state"])   # moves state to param device
    if scheduler is not None and ckpt["scheduler_state"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    _set_rng_states(ckpt["rng_states"])

    print(f"Resumed from {path} (completed epoch {ckpt['epoch']})")
    return ckpt

def peek_resume_epoch(config):
    """Last completed epoch in the resume checkpoint, or None if absent."""
    path = resume_path(config)
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return int(ckpt["epoch"])