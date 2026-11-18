"""Batch loops with exact sample weighting for incomplete accumulation groups."""

import math
import time

import torch

from .data import tensor_batch


def _accumulation_group_size(loader, batch_index: int, accumulation: int) -> int:
    first_sample = (batch_index // accumulation) * accumulation * loader.batch_size
    return min(accumulation * loader.batch_size, len(loader.dataset) - first_sample)


def run_epoch(model, loader, state: dict) -> dict:
    """Train when optimizer is supplied; validation never mutates model parameters."""
    is_training = state.get("optimizer") is not None
    model.train(is_training)
    totals = dict.fromkeys(("sst_numerator", "sst_denominator", "auxiliary_numerator", "auxiliary_denominator"), 0.0)
    started = time.perf_counter()
    if is_training:
        state["optimizer"].zero_grad(set_to_none=True)
    telemetry = state.setdefault("telemetry", {"updates": 0, "overflow_skips": 0})
    updates_before = telemetry["updates"]
    with torch.set_grad_enabled(is_training):
        for batch_index, source in enumerate(loader):
            batch = tensor_batch(source, state["device"])
            with torch.cuda.amp.autocast(enabled=state["amp"]):
                output = model(batch)
            losses = state["loss"](output, batch)
            if is_training:
                _backward_step(losses["total"], loader, dict(state, batch_index=batch_index,
                               batch_samples=batch["base"].shape[0]))
            for name in totals:
                totals[name] += float(losses[name].detach())
    if state["device"].type == "cuda":
        torch.cuda.synchronize()
    if is_training and telemetry["updates"] == updates_before:
        raise FloatingPointError("Epoch performed no finite optimizer updates")
    return _epoch_result(totals, {"elapsed": time.perf_counter() - started,
                                  "weight": state["loss"].auxiliary_weight, "samples": len(loader.dataset)})


def _backward_step(loss: torch.Tensor, loader, state: dict) -> None:
    accumulation = state["config"].accumulation_steps
    index = state["batch_index"]
    group_size = _accumulation_group_size(loader, index, accumulation)
    state["scaler"].scale(loss * state["batch_samples"] / group_size).backward()
    is_boundary = (index + 1) % accumulation == 0 or index + 1 == len(loader)
    if not is_boundary:
        return
    optimizer, scaler = state["optimizer"], state["scaler"]
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(state["model"].parameters(), state["config"].gradient_clip)
    if not torch.isfinite(norm):
        if not scaler.is_enabled():
            raise FloatingPointError("Nonfinite gradients; no optimizer update was applied")
        state["telemetry"]["overflow_skips"] += 1
        scaler.update(scaler.get_scale() / 2)
        optimizer.zero_grad(set_to_none=True)
        return
    scaler.step(optimizer)
    state["telemetry"]["updates"] += 1
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def _epoch_result(totals: dict, options: dict) -> dict:
    if totals["sst_denominator"] <= 0 or totals["auxiliary_denominator"] <= 0:
        raise ValueError("Training/validation split or selected subset has no valid samples")
    sst = totals["sst_numerator"] / totals["sst_denominator"]
    auxiliary = totals["auxiliary_numerator"] / totals["auxiliary_denominator"]
    if not math.isfinite(sst + auxiliary):
        raise FloatingPointError("Nonfinite aggregated epoch metrics")
    return {"sst_mse_C2": sst, "sst_rmse_C": math.sqrt(sst), "auxiliary": auxiliary,
            "total": sst + options["weight"] * auxiliary,
            "seconds": options["elapsed"], "samples": options["samples"]}
