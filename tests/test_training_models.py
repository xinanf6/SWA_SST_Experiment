"""CPU numerical checks for the C0 port and physical residual units."""

from types import SimpleNamespace

import numpy as np
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("tqdm")

from sst_training.epochs import run_epoch
from sst_training.losses import C0Loss
from sst_training.models.dspg import ModelSettings
from sst_training.models.physics import AdvectionDiffusionBlock
from sst_training.models.predictor import C0Predictor


TINY = ModelSettings(spatial_channels=8, global_hidden=8, pixel_hidden=4,
                     fusion_channels=8, decoder_channels=16, decoder_hidden=8)
STATISTICS = {"variables": {"sst": {"mean": 15., "std": 4.}}}


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _batch(count: int = 1) -> dict:
    generator = torch.Generator().manual_seed(9)
    return {"base": torch.randn(count, 14, 7, 7, 9, generator=generator) * .1,
            "thermal": torch.randn(count, 14, 3, 7, 9, generator=generator) * .1,
            "last_sst": torch.full((count, 1, 7, 9), 15.),
            "target": torch.zeros(count, 3, 1, 7, 9)}


def _context():
    mask = np.zeros((7, 9), bool)
    mask[1:-1, 1:-1] = True
    return SimpleNamespace(statistics=STATISTICS, grid={"core_mask": mask, "ocean_mask": mask,
                    "area_weights": np.broadcast_to(np.linspace(.7, .8, 7)[:, None], (7, 9))})


def test_c0_shapes_and_all_trainable_branches_receive_finite_gradients():
    model, batch = C0Predictor(TINY, STATISTICS), _batch()
    output = model(batch)
    assert output["predictions"].shape == (1, 3, 1, 7, 9)
    assert output["displacement_pixels"].shape == (1, 14, 2, 7, 9)
    losses = C0Loss(_context())(output, batch)
    losses["total"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_residual_uses_standard_deviation_without_adding_temperature_mean():
    model, batch = C0Predictor(TINY, STATISTICS), _batch()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.backbone.decoder[-1].bias.copy_(torch.tensor([1., 2., -1.]))
    output = model(batch)
    expected = torch.tensor([19., 23., 11.])
    torch.testing.assert_close(output["predictions"][0, :, 0, 3, 3], expected)
    torch.testing.assert_close(output["residual"][0, :, 0, 3, 3], expected - 15.)


def test_constant_sst_scale_matches_preprocessor_unit_scale():
    model = C0Predictor(TINY, {"variables": {"sst": {"mean": 15., "std": 0.}}})
    assert model.sst_scale.item() == 1.


def test_c0_loss_matches_hand_calculation_and_ignores_noncore_nan():
    batch = _batch()
    prediction = torch.full((1, 3, 1, 7, 9), 17., requires_grad=True)
    prediction = prediction.clone()
    prediction[..., 0, :] = float("nan")
    displacement = batch["base"][:, :, 3:5].clone() + 3.
    result = C0Loss(_context())({"predictions": prediction, "displacement_pixels": displacement}, batch)
    assert result["sst"].item() == pytest.approx(4.)
    assert result["auxiliary"].item() == pytest.approx(9.)
    assert result["total"].item() == pytest.approx(4.9)
    result["total"].backward()


def test_zero_displacement_and_zero_diffusion_is_identity():
    block = AdvectionDiffusionBlock(8)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
    inputs = torch.randn(2, 5, 7, 9)
    output, displacement = block(torch.randn(2, 8, 7, 9), inputs)
    torch.testing.assert_close(output, inputs[:, :1], atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(displacement) == 0


def test_backbone_initialization_is_isolated_from_thermal_module_rng():
    torch.manual_seed(2026)
    first = C0Predictor(TINY, STATISTICS)
    torch.manual_seed(2026)
    second = C0Predictor(TINY, STATISTICS)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name])


@pytest.mark.parametrize("name,shape", [("base", (1, 13, 7, 7, 9)),
        ("thermal", (1, 14, 2, 7, 9)), ("last_sst", (1, 7, 9))])
def test_model_rejects_wrong_history_and_unit_container_shapes(name, shape):
    batch = _batch()
    batch[name] = torch.zeros(shape)
    with pytest.raises(ValueError):
        C0Predictor(TINY, STATISTICS)(batch)


def test_uneven_accumulation_matches_one_full_batch_update():
    from torch.utils.data import DataLoader
    from sst_training.config import TrainingConfig
    from pathlib import Path
    batch = _batch(3)
    samples = [{key: value[index] for key, value in batch.items()} for index in range(3)]
    torch.manual_seed(8)
    first = C0Predictor(TINY, STATISTICS)
    second = C0Predictor(TINY, STATISTICS)
    second.load_state_dict(first.state_dict())
    for model, size, accumulation in ((first, 3, 1), (second, 2, 2)):
        config = TrainingConfig(Path("prepared"), Path("training"), batch_size=size,
                                accumulation_steps=accumulation, gradient_clip=100.)
        state = {"config": config, "device": torch.device("cpu"), "amp": False,
                 "optimizer": torch.optim.SGD(model.parameters(), lr=.01), "model": model,
                 "scaler": torch.cuda.amp.GradScaler(enabled=False), "loss": C0Loss(_context())}
        run_epoch(model, DataLoader(samples, batch_size=size), state)
        assert state["telemetry"]["updates"] == 1
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], atol=1e-6, rtol=1e-5)
