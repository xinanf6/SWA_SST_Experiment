"""Read-only batch progress; numerical training state is never modified here."""

from contextlib import contextmanager

from tqdm import tqdm


@contextmanager
def epoch_progress(loader, state: dict):
    """Always close the bar, including an interrupted or failed epoch."""
    stage = "Train" if state.get("optimizer") is not None else "Valid"
    epoch = state.get("epoch", 0) + 1
    description = "Epoch %d/%d %s" % (epoch, state["config"].max_epochs, stage)
    with tqdm(total=len(loader), desc=description, unit="batch", ascii=True,
              dynamic_ncols=True, mininterval=1.0, leave=True,
              disable=not state.get("progress_enabled", True)) as progress:
        yield progress


def update_progress(progress, totals: dict, state: dict) -> None:
    """Display the same accumulated sufficient statistics as the epoch summary."""
    optimizer = state.get("optimizer")
    learning_rate = optimizer.param_groups[0]["lr"] if optimizer is not None else state.get(
        "current_learning_rate", state["config"].learning_rate)
    if totals["sst_denominator"] > 0 and totals["auxiliary_denominator"] > 0:
        progress.set_postfix({"MSE": "%.6f" % (totals["sst_numerator"] / totals["sst_denominator"]),
                "aux": "%.4f" % (totals["auxiliary_numerator"] / totals["auxiliary_denominator"]),
                "lr": "%.2e" % learning_rate}, refresh=False)
    progress.update(1)
