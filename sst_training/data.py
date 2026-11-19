"""Torch adapter over immutable daily memmaps; no window expansion on disk."""

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from sst_preprocessing.dataset import PreparedSSTDataset


BASE_TENSOR_KEYS = ("base", "target", "last_sst")
TENSOR_KEYS = ("base", "thermal", "target", "last_sst")


class TorchSSTDataset(Dataset):
    """Read the requested input branches; targets remain normalized until loss."""

    def __init__(self, root: Path, split: str, *, has_thermal_input: bool = True) -> None:
        self.prepared = PreparedSSTDataset(root, split)
        self.has_thermal_input = has_thermal_input

    def __len__(self) -> int:
        return len(self.prepared)

    def __getitem__(self, position: int) -> dict:
        if self.has_thermal_input:
            source = self.prepared.read_sample(position, include_front=False, include_heat_increment=False)
        else:
            source = self.prepared.read_base_sample(position)
        tensors = {name: torch.from_numpy(source[name]) for name in _tensor_keys(source)}
        return dict(tensors, origin_date=source["origin_date"],
                    target_dates=source["target_dates"].astype("datetime64[D]").astype(str).tolist())

    def close(self) -> None:
        self.prepared.close()


def seed_worker(worker_id: int) -> None:
    """Independent worker streams remain deterministic after Windows spawn."""
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def build_loader(dataset: TorchSSTDataset, config, options: dict) -> DataLoader:
    selected = dataset
    if options.get("limit") is not None:
        selected = Subset(dataset, range(min(int(options["limit"]), len(dataset))))
    generator = torch.Generator().manual_seed(options.get("seed", config.seed))
    return DataLoader(selected, batch_size=config.batch_size, shuffle=options.get("shuffle", False),
                      num_workers=config.workers, pin_memory=options.get("cuda", False),
                      worker_init_fn=seed_worker, generator=generator, drop_last=False)


def tensor_batch(batch: dict, device: torch.device) -> dict:
    """Copy only model/loss tensors; dates stay on the host in canonical order."""
    return {name: batch[name].to(device, non_blocking=True) for name in _tensor_keys(batch)}


def _tensor_keys(batch: dict) -> tuple:
    return TENSOR_KEYS if "thermal" in batch else BASE_TENSOR_KEYS
