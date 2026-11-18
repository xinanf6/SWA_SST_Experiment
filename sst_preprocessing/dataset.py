"""Read completed daily memmaps and construct windows without storing duplicates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .constants import FEATURE_GROUPS
from .errors import DataValidationError
from .statistics import denormalize_values


class PreparedSSTDataset:
    """NumPy-only data access with an explicit one-day environmental lag.

    A sample has SST/front history t-13..t, environmental/thermal history
    t-14..t-1, normalized targets t+1..t+3, Celsius ``last_sst`` [1,H,W],
    and the original heat prior ``heat_increment`` [3,1,H,W]. Invalid cells
    are zero, including physical outputs; masks are available in metadata.
    """

    def __init__(self, output_root: Union[str, Path], split: str = "train") -> None:
        self.output_root = Path(output_root)
        if not (self.output_root / "READY.json").is_file():
            raise DataValidationError("Preprocessing output is incomplete: READY.json is missing")
        self.statistics = self._read_json("stats.json")
        self.ready = self._read_json("READY.json")
        self._validate_readiness()
        self.dates = np.load(self.output_root / "dates.npy", allow_pickle=False)
        with np.load(self.output_root / "grid_masks.npz", allow_pickle=False) as archive:
            self.metadata = {name: archive[name] for name in archive.files}
        index = pd.read_csv(self.output_root / "sample_index.csv")
        if split not in {"train", "validation", "test"}:
            raise DataValidationError("Unknown dataset split: %s" % split)
        self.sample_index = index.loc[index["split"] == split].reset_index(drop=True)
        self.ocean_mask = self._resolve_ocean_mask()
        self._open_arrays()

    def _read_json(self, filename: str) -> dict:
        with (self.output_root / filename).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _validate_readiness(self) -> None:
        if self.ready.get("status") != "complete":
            raise DataValidationError("Preprocessing output is incomplete: READY.json status must be complete")
        identity = self.ready.get("run_identity")
        if not isinstance(identity, str) or not identity.strip():
            raise DataValidationError("READY.json must contain a nonempty run_identity")
        if self.statistics.get("run_identity") != identity:
            raise DataValidationError("READY.json and stats.json have inconsistent run_identity")
        if (self.output_root / "run.json").is_file():
            if self._read_json("run.json").get("run_identity") != identity:
                raise DataValidationError("READY.json and run.json have inconsistent run_identity")

    def _open_arrays(self) -> None:
        """Side effect: open mappings, releasing every handle if any file is invalid."""
        self.arrays = {}
        try:
            for name in FEATURE_GROUPS:
                self.arrays[name] = np.load(self.output_root / "normalized" / (name + ".npy"),
                                             mmap_mode="r", allow_pickle=False)
            self._validate_arrays()
        except Exception:
            self.close()
            raise

    def __getstate__(self) -> dict:
        # Windows data-loader workers spawn: serializing memmaps would copy all daily data.
        state = self.__dict__.copy()
        state.pop("arrays")
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._open_arrays()

    def _resolve_ocean_mask(self) -> np.ndarray:
        for name in ("ocean_mask", "common_ocean_mask", "main_ocean_mask"):
            if name in self.metadata:
                return np.asarray(self.metadata[name], dtype=bool)
        raise DataValidationError("grid_masks.npz must contain an ocean_mask")

    def _validate_arrays(self) -> None:
        for group, variables in FEATURE_GROUPS.items():
            expected_shape = (len(self.dates), len(variables)) + self.ocean_mask.shape
            if self.arrays[group].shape != expected_shape:
                raise DataValidationError("Unexpected %s array shape: %s; expected %s" % (
                    group, self.arrays[group].shape, expected_shape))
            if self.arrays[group].dtype != np.float32:
                raise DataValidationError("Normalized %s array must use float32" % group)

    def __len__(self) -> int:
        return len(self.sample_index)

    def close(self) -> None:
        """Side effect: release read-only mappings before Windows moves output files."""
        for array in self.arrays.values():
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()
        self.arrays = {}

    def __enter__(self) -> "PreparedSSTDataset":
        if not self.arrays:
            raise DataValidationError("Dataset is closed; create a new PreparedSSTDataset")
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()

    def _physical_values(self, normalized: np.ndarray, variable: str) -> np.ndarray:
        statistics = self.statistics["variables"][variable]
        values = denormalize_values(normalized, statistics["mean"], statistics["std"])
        return np.where(self.ocean_mask, values, 0.0).astype(np.float32)

    def _masked_copy(self, values: np.ndarray, group: str) -> np.ndarray:
        mask = self.metadata.get("gradient_mask", self.ocean_mask) if group == "front" else self.ocean_mask
        return np.where(mask, values, np.float32(0))

    @staticmethod
    def _target_indices(row: pd.Series) -> list:
        columns = sorted(
            [name for name in row.index if name.startswith("target_") and name.endswith("_index")],
            key=lambda name: int(name.split("_")[1]),
        )
        return [int(row[name]) for name in columns]

    def __getitem__(self, sample_number: int) -> dict:
        """Return independent arrays so callers cannot mutate the shared memmaps."""
        return self.read_sample(sample_number)

    def read_sample(self, sample_number: int, *, include_front: bool = True,
                    include_heat_increment: bool = True) -> dict:
        """Materialize the shared inputs and only the requested optional branches."""
        if not self.arrays:
            raise DataValidationError("Dataset is closed; create a new PreparedSSTDataset")
        row = self.sample_index.iloc[sample_number]
        sst_slice = slice(int(row.sst_start_index), int(row.sst_end_index) + 1)
        environment_slice = slice(int(row.environment_start_index), int(row.environment_end_index) + 1)
        base = self._base_history(sst_slice, environment_slice)
        target_indices = self._target_indices(row)
        sample = {
            "base": base,
            "thermal": self._masked_copy(self.arrays["thermal"][environment_slice], "thermal"),
            "target": self._masked_copy(self.arrays["base"][target_indices, :1], "base"),
            "last_sst": self._physical_values(self.arrays["base"][int(row.origin_index), :1], "sst"),
            "origin_date": str(row.origin_date),
            "target_dates": self.dates[target_indices].copy(),
        }
        if include_front:
            sample["front"] = self._masked_copy(self.arrays["front"][sst_slice], "front")
        if include_heat_increment:
            sample["heat_increment"] = self._heat_prior(int(row.environment_end_index), len(target_indices))
        return sample

    def _base_history(self, sst_slice: slice, environment_slice: slice) -> np.ndarray:
        values = self.arrays["base"]
        base = np.empty_like(values[sst_slice], subok=False)
        base[:, :1] = values[sst_slice, :1]
        base[:, 1:] = values[environment_slice, 1:]
        np.copyto(base, np.float32(0), where=~self.ocean_mask)
        return base

    def _heat_prior(self, environment_end: int, lead_count: int) -> np.ndarray:
        heat = self._physical_values(self.arrays["thermal"][environment_end, -1], "heat_increment")
        leads = np.arange(1, lead_count + 1, dtype=np.float32)[:, None, None, None]
        return leads * heat[None, None]
