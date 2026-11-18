"""Progress formatting, exception closure and audited checkpoint compatibility."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("tqdm")

from sst_training.progress import epoch_progress, update_progress
from sst_training.runtime import code_fingerprint, compatible_code_hashes


LEGACY_CODE = "eb8e3f3d5c9a847be8bb61e8f5095916a8258f23c0fd08ac925dffa07278a31a"


def _state(training: bool = True) -> dict:
    return {"config": SimpleNamespace(max_epochs=50, learning_rate=.001), "epoch": 2,
            "optimizer": SimpleNamespace(param_groups=[{"lr": .003}]) if training else None,
            "current_learning_rate": .0005}


def _totals() -> dict:
    return {"sst_numerator": 4., "sst_denominator": 2.,
            "auxiliary_numerator": 1., "auxiliary_denominator": 2.}


def test_train_bar_displays_batches_cumulative_mse_and_actual_learning_rate(capsys):
    state = _state()
    with epoch_progress([0, 1], state) as progress:
        for _ in range(2):
            update_progress(progress, _totals(), state)
    output = capsys.readouterr().err
    assert "Epoch 3/50 Train" in output and "2/2" in output
    assert "MSE=2.000000" in output and "lr=3.00e-03" in output
    assert "batch/s" in output or "s/batch" in output


def test_validation_bar_uses_validation_stage_and_current_learning_rate(capsys):
    state = _state(False)
    with epoch_progress([0], state) as progress:
        update_progress(progress, _totals(), state)
    output = capsys.readouterr().err
    assert "Valid" in output and "1/1" in output
    assert "lr=5.00e-04" in output


def test_progress_can_be_disabled_without_output(capsys):
    state = dict(_state(), progress_enabled=False)
    with epoch_progress([0], state) as progress:
        update_progress(progress, _totals(), state)
    assert capsys.readouterr().err == ""


def test_failed_batch_closes_bar_without_claiming_completion(capsys):
    state = _state()
    with pytest.raises(RuntimeError, match="failed batch"):
        with epoch_progress([0, 1], state) as progress:
            update_progress(progress, _totals(), state)
            raise RuntimeError("failed batch")
    assert progress.disable and progress.n == 1
    assert "2/2" not in capsys.readouterr().err


def test_only_pinned_display_release_is_compatible_with_legacy_code():
    current = code_fingerprint()
    assert compatible_code_hashes(LEGACY_CODE, current)
    assert compatible_code_hashes(current, LEGACY_CODE)
    assert compatible_code_hashes(current, current)
    assert not compatible_code_hashes("unknown-training-code", current)
    assert not compatible_code_hashes(LEGACY_CODE, "future-unreviewed-code")
