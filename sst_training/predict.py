"""Stream a trained C0 checkpoint into the existing full-split forecast contract."""

from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import torch

from sst_analysis.config import validate_output_location
from sst_analysis.context import load_context
from sst_preprocessing.errors import DataValidationError

from .data import TorchSSTDataset, build_loader, tensor_batch
from .models.dspg import ModelSettings
from .models.predictor import C0Predictor
from .runtime import resolve_device


DATE_UNITS = "days since 1970-01-01 00:00:00"


def export_predictions(config, options: dict) -> dict:
    """Side effect: write all canonical origins; never accepts an untrained model."""
    context = load_context(config.prepared_root)
    output = Path(options["output"]).resolve()
    if output == Path(options["checkpoint"]).resolve():
        raise DataValidationError("Prediction output must not overwrite its checkpoint")
    validate_output_location(output.parent, context.root)
    validate_output_location(output.parent, Path(context.run["config"]["data_root"]))
    if output.suffix != ".nc" or output.exists() and not options.get("overwrite", False):
        raise DataValidationError("Prediction output must be a new .nc file or explicitly overwritten")
    saved = torch.load(Path(options["checkpoint"]), map_location="cpu")
    _validate_checkpoint(saved, context.run_identity, config)
    if saved["metadata"]["kind"] == "smoke" and not options.get("allow_smoke_checkpoint", False):
        raise DataValidationError("Smoke checkpoints require --allow-smoke-checkpoint for explicit debug export")
    device = resolve_device(config.device)
    model = C0Predictor(ModelSettings(**config.architecture), context.statistics).to(device).eval()
    model.load_state_dict(saved["model"])
    dataset = TorchSSTDataset(context.root, options.get("split", "validation"))
    temporary = output.with_name(output.name + ".tmp")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_predictions(model, dataset, {"config": config, "context": context,
                    "device": device, "path": temporary, "metadata": saved["metadata"]})
        temporary.replace(output)
    finally:
        dataset.close()
        if temporary.exists():
            temporary.unlink()
    return {"status": "complete", "origins": len(dataset), "output": str(output), "run_identity": context.run_identity,
            "checkpoint_epoch": saved["epoch"], "training_kind": saved["metadata"]["kind"]}


def _validate_checkpoint(saved: dict, identity: str, config) -> None:
    metadata = saved.get("metadata", {})
    if metadata.get("run_identity") != identity or metadata.get("config_fingerprint") != config.fingerprint():
        raise DataValidationError("Checkpoint data identity or training configuration does not match")
    if saved.get("epoch", 0) < 1 or metadata.get("model") != "C0":
        raise DataValidationError("Prediction requires a trained C0 epoch checkpoint")


def _create_file(handle, dataset: TorchSSTDataset, context) -> tuple:
    grid, count = context.grid, len(dataset)
    for name, size in (("origin", count), ("lead", 3), ("latitude", len(grid["latitude"])),
                       ("longitude", len(grid["longitude"]))):
        handle.createDimension(name, size)
    origins = handle.createVariable("origin", "i4", ("origin",))
    targets = handle.createVariable("target_dates", "i4", ("origin", "lead"))
    for variable in (origins, targets):
        variable.units, variable.calendar = DATE_UNITS, "proleptic_gregorian"
    for name in ("latitude", "longitude"):
        handle.createVariable(name, "f8", (name,))[:] = grid[name]
    handle.createVariable("lead", "i4", ("lead",))[:] = [1, 2, 3]
    dims = ("origin", "lead", "latitude", "longitude")
    prediction = handle.createVariable("predictions", "f4", dims, zlib=True, complevel=1,
                    chunksizes=(1, 1, len(grid["latitude"]), len(grid["longitude"])), fill_value=np.nan)
    residual = handle.createVariable("residual", "f4", dims, zlib=True, complevel=1,
                    chunksizes=(1, 1, len(grid["latitude"]), len(grid["longitude"])))
    last_sst = handle.createVariable("last_sst", "f4", ("origin", "latitude", "longitude"), zlib=True, complevel=1,
                    chunksizes=(1, len(grid["latitude"]), len(grid["longitude"])))
    prediction.units = residual.units = "degree_Celsius"
    last_sst.units = "degree_Celsius"
    handle.run_identity = context.run_identity
    return prediction, residual, last_sst, origins, targets


def _write_predictions(model, dataset: TorchSSTDataset, state: dict) -> None:
    if not len(dataset):
        raise DataValidationError("Cannot export predictions for an empty split")
    loader = build_loader(dataset, state["config"], {"cuda": state["device"].type == "cuda"})
    ocean = state["context"].grid["ocean_mask"]
    with netCDF4.Dataset(str(state["path"]), "w") as handle:
        prediction, residual, last_sst, origins, targets = _create_file(handle, dataset, state["context"])
        handle.training_kind = state["metadata"]["kind"]
        handle.model_name = "C0"
        position = 0
        with torch.no_grad():
            for batch in loader:
                with torch.cuda.amp.autocast(enabled=state["config"].amp and state["device"].type == "cuda"):
                    output = model(tensor_batch(batch, state["device"]))
                count = output["predictions"].shape[0]
                selected = slice(position, position + count)
                selection = {"slice": selected, "ocean": ocean}
                for name, variable in (("predictions", prediction), ("residual", residual)):
                    _store_temperature(variable, output[name][:, :, 0], selection)
                _store_temperature(last_sst, batch["last_sst"][:, 0], selection)
                origins[selected] = np.asarray(batch["origin_date"], dtype="datetime64[D]").astype("i4")
                targets[selected] = np.asarray(batch["target_dates"], dtype="datetime64[D]").T.astype("i4")
                position += count


def _store_temperature(variable, tensor: torch.Tensor, selection: dict) -> None:
    values = tensor.float().cpu().numpy()
    if not np.isfinite(values[..., selection["ocean"]]).all():
        raise FloatingPointError("Nonfinite forecast/decomposition on the fixed ocean domain")
    variable[selection["slice"]] = np.where(selection["ocean"], values, np.nan)
