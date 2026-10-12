"""Numerical regressions for the paper's spatial quantization contract."""

import pytest
import torch

from mcaq_yolo.core.quantization import QuantizationParameters, SpatialAdaptiveQuantization
from mcaq_yolo.core.morphology import MorphologicalComplexityAnalyzer


@pytest.mark.parametrize("value", [2.0, -2.0, 0.0])
def test_constant_channels_are_not_clipped_to_zero(value):
    q = SpatialAdaptiveQuantization(smooth_transitions=False).eval()
    x = torch.full((1, 2, 4, 4), value)
    actual = q(x, torch.full((1, 2, 2), 4.0))
    torch.testing.assert_close(actual, x)


def test_affine_quantizer_represents_zero_exactly():
    params = QuantizationParameters(4)
    scale, zp = params.compute_scale_zeropoint(torch.tensor(-0.7), torch.tensor(1.0))
    assert zp == zp.round()
    q = SpatialAdaptiveQuantization(smooth_transitions=False).eval()
    x = torch.tensor([-0.7, 0.0, 1.0]).reshape(1, 1, 1, 3)
    y = q(x, torch.full((1, 1, 1), 4.0))
    assert y[0, 0, 0, 1] == 0
    assert (y - x).abs().max() <= scale / 2 + 1e-6


@pytest.mark.parametrize("bits", [1.0, 9.0, float("nan"), float("inf")])
def test_invalid_bit_values_rejected_before_quantization(bits):
    q = SpatialAdaptiveQuantization(smooth_transitions=False).eval()
    with pytest.raises(ValueError, match="bit_map"):
        q(torch.ones(1, 1, 4, 4), torch.full((1, 2, 2), bits))


@pytest.mark.parametrize("shape", [(0, 2, 2), (1, 0, 2), (1, 2), (2, 2, 2)])
def test_invalid_bit_map_shape_rejected(shape):
    q = SpatialAdaptiveQuantization(smooth_transitions=False).eval()
    with pytest.raises(ValueError, match="bit_map"):
        q(torch.ones(1, 1, 4, 4), torch.full(shape, 4.0))


@pytest.mark.parametrize("backend", ["gpu", "cv2"])
@pytest.mark.parametrize("shape,expected", [((80, 80), (8, 8)), ((20, 20), (8, 8)), ((19, 27), (8, 8)), ((3, 5), (3, 5))])
def test_grid_size_means_actual_tile_count(backend, shape, expected):
    analyzer = MorphologicalComplexityAnalyzer(device="cpu", grid_size=8, metric_backend=backend)
    phi, _ = analyzer.compute_phi_tiles(torch.rand(1, 2, *shape))
    assert phi.shape == (1, *expected, 8)
    assert torch.isfinite(phi).all()
    assert phi.min() >= 0 and phi.max() <= 1 + 1e-5


@pytest.mark.parametrize("backend", ["gpu", "cv2"])
def test_last_pixels_contribute_to_last_tile(backend):
    analyzer = MorphologicalComplexityAnalyzer(device="cpu", grid_size=4, metric_backend=backend)
    x = torch.zeros(1, 1, 19, 27)
    x[:, :, 0, 0] = 1  # keep the image normalization unchanged
    before, _ = analyzer.compute_phi_tiles(x)
    x[:, :, -1, -1] = 1
    after, _ = analyzer.compute_phi_tiles(x)
    assert not torch.allclose(before[:, -1, -1], after[:, -1, -1])


def test_calibration_histogram_roundtrips_in_checkpoint():
    source = SpatialAdaptiveQuantization(calibration_mode="entropy", smooth_transitions=False)
    source.update_running_stats(torch.randn(1, 2, 4, 4))
    restored = SpatialAdaptiveQuantization(calibration_mode="entropy", smooth_transitions=False)
    restored.load_state_dict(source.state_dict())
    torch.testing.assert_close(restored.calibration_histogram, source.calibration_histogram)


@pytest.fixture(scope="module")
def detector():
    from mcaq_yolo.models.mcaq_yolo import MCAQYOLO
    torch.manual_seed(7)
    return MCAQYOLO(model_name="yolov8n.yaml", pretrained=False, device="cpu").eval()


def test_precomputed_maps_match_online_inference_and_skip_analyzer(detector, monkeypatch):
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        expected, aux = detector(x)
        maps = detector.precompute_bit_maps(x)
        for first, second in zip(maps, aux['bit_map']):
            torch.testing.assert_close(first, second)

        def forbid_analysis(*args, **kwargs):
            raise AssertionError("precomputed path must not recompute morphology")

        monkeypatch.setattr(detector.complexity_analyzer, "forward", forbid_analysis)
        actual, reused = detector(x, precomputed_bit_maps=maps)
        torch.testing.assert_close(actual[0], expected[0])
        assert reused['complexity_map'] == [None, None, None]


def test_precomputed_maps_reject_training_and_missing_scales(detector):
    x = torch.rand(1, 3, 64, 64)
    with pytest.raises(ValueError, match="scale"):
        detector(x, precomputed_bit_maps=[])
    detector.train()
    try:
        with pytest.raises(ValueError, match="inference"):
            detector(x, precomputed_bit_maps=[torch.full((1, 8, 8), 4.0)] * 3)
    finally:
        detector.eval()


def test_recalibration_refreshes_frozen_stats_and_honors_image_limit(detector):
    observed = []
    handle = detector.model.register_forward_pre_hook(lambda module, args: observed.append(args[0].shape[0]))
    try:
        seen = detector.calibrate([{'img': torch.rand(3, 3, 64, 64)}], num_images=2)
        assert seen == 2 and observed == [2]
        first = [q.running_max.clone() for q in detector.quantizers.values()]
        detector.calibrate([{'img': torch.zeros(1, 3, 64, 64)}], num_images=1)
        assert all(int(q.num_batches_tracked) == 1 for q in detector.quantizers.values())
        assert any(not torch.equal(old, q.running_max) for old, q in zip(first, detector.quantizers.values()))
    finally:
        handle.remove()


def test_empty_calibration_is_not_reported_as_frozen(detector):
    with pytest.raises(ValueError, match="empty"):
        detector.calibrate([], num_images=1)
    assert all(not bool(q.stats_frozen) for q in detector.quantizers.values())


def test_fractal_backends_count_partial_boxes_consistently():
    analyzer = MorphologicalComplexityAnalyzer(device="cpu")
    edge = torch.ones(1, 1, 10, 14)
    gpu = analyzer._fractal_dimension_tiles(edge, (10, 14)).item()
    cv2 = analyzer.fast_fractal_dimension(edge[0, 0].numpy())
    assert gpu == pytest.approx(cv2, abs=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("per_channel", [False, True])
def test_quantizer_preserves_activation_dtype(dtype, per_channel):
    q = SpatialAdaptiveQuantization(smooth_transitions=False, per_channel=per_channel).eval()
    x = torch.linspace(-1, 1, 40, dtype=dtype).reshape(1, 1, 5, 8)
    y = q(x, torch.full((1, 2, 3), 4.0))
    assert y.dtype == dtype and torch.isfinite(y).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_zero_tensor_stays_finite(dtype):
    q = SpatialAdaptiveQuantization(smooth_transitions=False, per_channel=False).eval()
    x = torch.zeros(1, 2, 4, 4, dtype=dtype)
    torch.testing.assert_close(q(x, torch.full((1, 2, 2), 4.)), x)


def test_constant_nonzero_tiles_have_no_artificial_edges():
    x = torch.ones(1, 1, 80, 80)
    x[:, :, :40] = 0
    analyzer = MorphologicalComplexityAnalyzer(device="cpu", grid_size=8)
    _, metrics = analyzer.compute_phi_tiles(x)
    assert metrics['gradient'][0, 6, 6] == pytest.approx(0., abs=1e-6)
    assert metrics['edge'][0, 6, 6] == 0


@pytest.mark.parametrize("empty", [False, True])
def test_calibration_restores_mixed_module_modes(detector, empty):
    detector.train()
    bn = next(m for m in detector.model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    bn.eval()
    modes = {m: m.training for m in detector.modules()}
    try:
        if empty:
            with pytest.raises(ValueError, match="empty"):
                detector.calibrate([], num_images=1)
        else:
            detector.calibrate([torch.rand(1, 3, 64, 64)], num_images=1)
        assert all(m.training == flag for m, flag in modes.items())
    finally:
        detector.eval()


@pytest.mark.parametrize("old_cache", [True, False])
def test_curriculum_scores_invalidate_old_layout_and_changed_grid(tmp_path, old_cache):
    import hashlib
    import json
    import numpy as np
    from types import SimpleNamespace
    from mcaq_yolo.train import Trainer

    class Images(torch.utils.data.Dataset):
        im_files = [str(tmp_path / 'image.png')]

        def __len__(self):
            return 1

        def __getitem__(self, index):
            torch.manual_seed(19)
            return {'img': torch.rand(3, 32, 32)}

    trainer = Trainer.__new__(Trainer)
    trainer.config = {}
    analyzer = MorphologicalComplexityAnalyzer(device='cpu', grid_size=4)
    trainer.model = SimpleNamespace(complexity_analyzer=analyzer)
    trainer.train_dataset = Images()
    trainer._build_scoring_dataset = lambda: trainer.train_dataset
    trainer.imgsz, trainer.device, trainer.batch_size = 32, 'cpu', 1
    trainer.complexity_path = tmp_path / 'scores.npy'
    if old_cache:
        meta = {'version': 2, 'augment': False, 'backend': 'gpu', 'imgsz': 32,
                'n': 1, 'files_md5': hashlib.md5(Images.im_files[0].encode()).hexdigest()}
        trainer.complexity_path.with_suffix('.meta.json').write_text(json.dumps(meta))
    else:
        trainer._compute_complexity_scores()
        analyzer.grid_size = 8
    # A recognizable stale value must never become the curriculum score.
    np.save(trainer.complexity_path, np.array([0.9999], dtype=np.float32))
    actual = trainer._compute_complexity_scores()
    expected = analyzer.score_image(Images()[0]['img'].unsqueeze(0))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape,grid", [((10, 13), (4, 5)), ((3, 5), (6, 8))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cuda_nondivisible_grid_and_dtype_parity(shape, grid, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime unavailable")
    pytest.importorskip("mcaq_cuda_ops")
    q = SpatialAdaptiveQuantization(smooth_transitions=False).cuda().eval()
    x = torch.linspace(-1, 2, shape[0] * shape[1], device="cuda", dtype=dtype).reshape(1, 1, *shape)
    bits = (torch.arange(grid[0] * grid[1], device="cuda").reshape(1, *grid) % 7 + 2).float()
    expected = q._forward_pytorch(x, bits, training=False)
    actual = q(x, bits)
    torch.testing.assert_close(actual, expected)


def test_cuda_rounding_ties_match_torch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime unavailable")
    pytest.importorskip("mcaq_cuda_ops")
    q = SpatialAdaptiveQuantization(smooth_transitions=False).cuda().eval()
    q.update_running_stats(torch.tensor([0., 15.], device="cuda").reshape(1, 1, 1, 2))
    q.freeze_calibration()
    x = torch.tensor([0.5, 1.5, 2.5, 3.5], device="cuda").reshape(1, 1, 1, 4)
    bits = torch.full((1, 1, 1), 4., device="cuda")
    torch.testing.assert_close(q(x, bits), torch.tensor([0., 2., 2., 4.], device="cuda").reshape_as(x))
