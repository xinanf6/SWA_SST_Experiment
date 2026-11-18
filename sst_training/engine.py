"""C0 training orchestration; best weights always use validation Celsius MSE."""

from dataclasses import asdict
import json
from pathlib import Path

import torch

from sst_analysis.context import load_context
from sst_analysis.config import validate_output_location
from sst_preprocessing.artifacts import read_json, write_json_atomic
from sst_preprocessing.errors import DataValidationError

from .config import validate_training_config
from .data import TorchSSTDataset, build_loader
from .epochs import run_epoch
from .losses import C0Loss
from .models.dspg import ModelSettings
from .models.predictor import C0Predictor
from .runtime import (code_fingerprint, compatible_code_hashes, resolve_device, restore_rng, rng_state,
                      save_checkpoint, seed_everything)


def _run_metadata(context, config, options: dict) -> dict:
    limits = {name: options.get(name) for name in ("train_limit", "validation_limit")}
    device = resolve_device(config.device)
    return {"model": "C0", "run_identity": context.run_identity,
            "config_fingerprint": config.fingerprint(), "training_code_sha256": code_fingerprint(),
            "config": config.to_dict(), "architecture": asdict(ModelSettings(**config.architecture)),
            "kind": options.get("kind", "full_training"), "limits": limits,
            "software": {"torch": torch.__version__, "cuda_runtime": torch.version.cuda},
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dspg_origin": read_json(Path(__file__).parent / "models" / "dspg_provenance.json"),
            "auxiliary_semantics": "Inherited learned pixel displacement vs standardized current; not SI velocity"}


def _prepare_output(root: Path, metadata: dict, resume: bool) -> None:
    if (root / "run.json").exists():
        if not resume:
            raise DataValidationError("Training output already exists; use --resume or another output directory")
        saved = read_json(root / "run.json")
        for name in ("run_identity", "config_fingerprint", "training_code_sha256", "kind", "limits"):
            if name == "training_code_sha256" and compatible_code_hashes(saved.get(name), metadata[name]):
                continue
            if saved.get(name) != metadata[name]:
                raise DataValidationError("Training resume mismatch for %s" % name)
        return
    if root.exists() and any(root.iterdir()):
        raise DataValidationError("Training output is nonempty without run.json")
    if resume:
        raise DataValidationError("Cannot resume: training run.json is missing")
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / "run.json", metadata)


def _training_state(context, config) -> dict:
    device = resolve_device(config.device)
    model = C0Predictor(ModelSettings(**config.architecture), context.statistics).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=config.scheduler_patience)
    has_amp = bool(config.amp and device.type == "cuda")
    return {"device": device, "model": model, "optimizer": optimizer, "scheduler": scheduler,
            "scaler": torch.cuda.amp.GradScaler(enabled=has_amp), "amp": has_amp,
            "loss": C0Loss(context, config.auxiliary_weight).to(device), "config": config,
            "epoch": 0, "best_mse": None, "bad_epochs": 0, "history": [],
            "best_model": None, "best_epoch": 0, "telemetry": {"updates": 0, "overflow_skips": 0}}


def _resume_state(root: Path, state: dict) -> None:
    if not (root / "last.pt").is_file():
        raise DataValidationError("Cannot resume: last.pt is missing")
    saved = torch.load(root / "last.pt", map_location="cpu")
    expected = read_json(root / "run.json")
    for name in ("run_identity", "config_fingerprint", "training_code_sha256", "kind", "limits"):
        if name == "training_code_sha256" and compatible_code_hashes(saved.get("metadata", {}).get(name), expected[name]):
            continue
        if saved.get("metadata", {}).get(name) != expected[name]:
            raise DataValidationError("Checkpoint metadata mismatch for %s" % name)
    for name in ("model", "optimizer", "scheduler", "scaler"):
        state[name].load_state_dict(saved[name])
    for name in ("epoch", "best_mse", "bad_epochs", "history", "best_model", "best_epoch", "telemetry"):
        state[name] = saved[name]
    # last.pt is authoritative if interruption happened between publishing the two files.
    save_checkpoint(root / "best.pt", _best_checkpoint(state, saved["metadata"]))
    restore_rng(saved["rng"])


def run_training(config, options: dict = None) -> dict:
    """Side effect: train only C0, publish epoch checkpoints and a truthful run status."""
    options = options or {}
    _validate_options(options)
    validate_training_config(config)
    context = load_context(config.prepared_root)
    validate_output_location(config.output_root, Path(context.run["config"]["data_root"]))
    metadata = _run_metadata(context, config, options)
    _prepare_output(config.output_root, metadata, options.get("resume", False))
    seed_everything(config.seed)
    state = _training_state(context, config)
    state["progress_enabled"] = options.get("progress", True)
    if options.get("resume"):
        _resume_state(config.output_root, state)
    training = TorchSSTDataset(context.root, "train")
    try:
        validation = TorchSSTDataset(context.root, "validation")
        try:
            _fit_epochs(state, {"training": training, "validation": validation}, dict(options, metadata=metadata))
        finally:
            validation.close()
    finally:
        training.close()
    summary = _training_summary(state, metadata)
    write_json_atomic(config.output_root / "training_summary.json", summary)
    return summary


def _validate_options(options: dict) -> None:
    for name in ("epochs", "train_limit", "validation_limit"):
        if name in options and (type(options[name]) is not int or options[name] < 1):
            raise DataValidationError("Training option %s must be a positive integer" % name)
    if options.get("kind", "full_training") not in {"full_training", "smoke"}:
        raise DataValidationError("Training kind must be full_training or smoke")


def _training_summary(state: dict, metadata: dict) -> dict:
    config = state["config"]
    has_finished = state["epoch"] >= config.max_epochs or state["bad_epochs"] >= config.early_stopping_patience
    status = "training_complete" if has_finished else "epoch_checkpoint"
    if metadata["kind"] == "smoke":
        status = "smoke_complete"
    memory = torch.cuda.max_memory_allocated(state["device"]) / (1024 ** 3) if state["device"].type == "cuda" else 0.
    return {"status": status, "run_identity": metadata["run_identity"], "epochs_completed": state["epoch"],
            "best_validation_mse_C2": state["best_mse"], "kind": metadata["kind"], "limits": metadata["limits"],
            "parameter_count": sum(value.numel() for value in state["model"].parameters()),
            "device": str(state["device"]), "peak_allocated_GiB": memory,
            "telemetry": state.get("telemetry", {}), "history": state["history"]}


def _fit_epochs(state: dict, datasets: dict, options: dict) -> None:
    config = state["config"]
    stop = min(config.max_epochs, state["epoch"] + options.get("epochs", config.max_epochs))
    for epoch in range(state["epoch"], stop):
        if state["bad_epochs"] >= config.early_stopping_patience:
            break
        loader = build_loader(datasets["training"], config, {"shuffle": True,
                    "limit": options.get("train_limit"), "seed": config.seed + epoch, "cuda": state["amp"]})
        training = run_epoch(state["model"], loader, state)
        validation_loader = build_loader(datasets["validation"], config,
                    {"limit": options.get("validation_limit"), "cuda": state["amp"]})
        validation = run_epoch(state["model"], validation_loader, dict(state, optimizer=None,
                               current_learning_rate=state["optimizer"].param_groups[0]["lr"]))
        _finish_epoch(state, {"training": training, "validation": validation}, options["metadata"])
        print("Epoch %d: train MSE %.6f, validation MSE %.6f" %
              (state["epoch"], training["sst_mse_C2"], validation["sst_mse_C2"]), flush=True)


def _finish_epoch(state: dict, metrics: dict, metadata: dict) -> None:
    score = metrics["validation"]["sst_mse_C2"]
    has_improved = state["best_mse"] is None or score < state["best_mse"]
    state["best_mse"] = score if has_improved else state["best_mse"]
    state["bad_epochs"] = 0 if has_improved else state["bad_epochs"] + 1
    state["epoch"] += 1
    if has_improved:
        state["best_model"] = {name: value.detach().cpu().clone()
                               for name, value in state["model"].state_dict().items()}
        state["best_epoch"] = state["epoch"]
    state["scheduler"].step(score)
    state["history"].append(dict(metrics, epoch=state["epoch"], learning_rate=state["optimizer"].param_groups[0]["lr"]))
    contents = {name: state[name].state_dict() for name in ("model", "optimizer", "scheduler", "scaler")}
    contents.update({name: state[name] for name in ("epoch", "best_mse", "bad_epochs", "history", "best_model", "best_epoch")})
    contents.update(rng=rng_state(), metadata=metadata)
    contents["telemetry"] = dict(state["telemetry"])
    root = state["config"].output_root
    save_checkpoint(root / "last.pt", contents)
    if has_improved:
        save_checkpoint(root / "best.pt", _best_checkpoint(state, metadata))
    write_json_atomic(root / "history.json", {"epochs": state["history"]})


def _best_checkpoint(state: dict, metadata: dict) -> dict:
    return {"model": state["best_model"], "epoch": state["best_epoch"],
            "best_mse": state["best_mse"], "metadata": metadata,
            "history": state["history"][:state["best_epoch"]], "checkpoint_role": "inference_best"}
