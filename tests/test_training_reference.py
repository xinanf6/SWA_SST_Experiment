"""Optional local DSPG parity audit; the portable suite does not need old sources."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pytest
torch = pytest.importorskip("torch")

from sst_training.models.dspg import DSPGBackbone, ModelSettings


def test_port_preserves_original_dspg_predictions_and_displacements(monkeypatch):
    path = Path(os.environ.get("DSPG_REFERENCE_PATH", "H:/fxa/DSPG/DSPG_code/model.py"))
    if not path.is_file():
        pytest.skip("Set DSPG_REFERENCE_PATH to run the optional original-source parity audit")
    provenance = Path(__file__).resolve().parents[1] / "sst_training/models/dspg_provenance.json"
    expected = json.loads(provenance.read_text())["source_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    settings = ModelSettings(spatial_channels=8, global_hidden=8, pixel_hidden=4, fusion_channels=8)
    stub = types.ModuleType("config")
    stub.PHYSICS_SCALE, stub.COMBINED_FEATURE_DIM = 5., 9
    monkeypatch.setitem(sys.modules, "config", stub)
    spec = importlib.util.spec_from_file_location("dspg_reference_audit", path)
    original_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_module)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _check_parity(original_module, settings)
    finally:
        torch.set_num_threads(previous)


def _check_parity(module, settings):
    original = module.DualStream_BiLSTM_Model(7, 8, 8, 4, 1, 3, (7, 9), 8).eval()
    port = DSPGBackbone(settings).eval()
    port.load_state_dict(original.state_dict(), strict=True)
    values = torch.randn(2, 3, 7, 7, 9)
    with torch.no_grad():
        expected, expected_displacement = original(values)
        actual, displacement = port(values, port.spatial_encoder(values))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(displacement, expected_displacement, atol=1e-6, rtol=1e-6)
