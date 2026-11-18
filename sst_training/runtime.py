"""Reproducible runtime, code identity and atomic training checkpoints."""

import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    selected = "cuda" if name == "auto" and torch.cuda.is_available() else name
    return torch.device("cpu" if selected == "auto" else selected)


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        digest.update(str(path.relative_to(Path(__file__).parent)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compatible_code_hashes(saved: str, expected: str) -> bool:
    """Only explicitly audited display-only releases may share an old checkpoint."""
    if saved == expected:
        return True
    path = Path(__file__).parent / "checkpoint_compatibility.json"
    with path.open(encoding="utf-8") as handle:
        groups = json.load(handle)["groups"]
    current = code_fingerprint()
    return any(current in group["hashes"] and saved in group["hashes"] and expected in group["hashes"]
               for group in groups)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, contents: dict) -> None:
    """Side effect: publish a complete checkpoint after the epoch boundary."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(contents, temporary)
    os.replace(str(temporary), str(path))
