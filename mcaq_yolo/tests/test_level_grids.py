"""Feature-scale grids preserve descriptor support without changing weights."""
import json

import pytest
import torch
import yaml

from mcaq_yolo.core.morphology import MorphologicalComplexityAnalyzer
from mcaq_yolo.models.mcaq_yolo import MCAQYOLO


def detector(**kwargs):
    return MCAQYOLO(model_name='yolov8n.yaml', pretrained=False, device='cpu', **kwargs)


def test_default_grids_keep_ten_pixel_tiles_at_640_and_backpropagate():
    model = detector().eval()
    with torch.no_grad():
        _, aux = model(torch.rand(1, 3, 640, 640))
    assert model.feature_grid_sizes == (8, 4, 2)
    assert [tuple(x.shape[-2:]) for x in aux['quantized_features']] == [(80, 80), (40, 40), (20, 20)]
    assert [tuple(x.shape[-2:]) for x in aux['bit_map']] == [(8, 8), (4, 4), (2, 2)]
    assert [tuple(x.shape[-2:]) for x in aux['complexity_map']] == [(8, 8), (4, 4), (2, 2)]
    model.train()
    _, train_aux = model(torch.rand(2, 3, 64, 64), precision_progress=1)
    loss = train_aux['avg_bits'] + sum(x.square().mean() for x in train_aux['quantized_features'])
    loss.backward()
    for module in (model.complexity_analyzer.complexity_mlp, model.bit_mapper, model.model.model[0]):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize('backend', ['gpu', 'cv2'])
def test_per_call_grid_covers_rectangular_and_tiny_inputs_without_mutation(backend):
    analyzer = MorphologicalComplexityAnalyzer(device='cpu', grid_size=8, metric_backend=backend)
    keys = tuple(analyzer.state_dict())
    for height, width, grid in [(21, 13, 4), (1, 3, 4), (20, 20, 2)]:
        complexity, descriptors = analyzer(torch.rand(1, 3, height, width), True, grid_size=grid)
        shape = (1, min(height, grid), min(width, grid))
        assert complexity.shape == shape and torch.isfinite(complexity).all()
        assert all(v.shape == shape and torch.isfinite(v).all() for v in descriptors.values())
    assert analyzer.grid_size == 8 and tuple(analyzer.state_dict()) == keys
    assert analyzer.compute_phi_tiles(torch.rand(1, 3, 20, 20))[0].shape == (1, 8, 8, 8)


def test_c5_structured_fixture_avoids_forced_fractal_fallback():
    # A flat quadrant plus textured quadrants: a concrete fixture, not a claim
    # that every feature map must yield diverse descriptors.
    features = torch.rand(1, 1, 20, 20, generator=torch.Generator().manual_seed(21))
    features[:, :, :10, :10] = 0
    analyzer = MorphologicalComplexityAnalyzer(device='cpu')
    old = analyzer.compute_phi_tiles(features, grid_size=8)[1]['fractal']
    new = analyzer.compute_phi_tiles(features, grid_size=2)[1]['fractal']
    assert torch.all(old == .5)
    assert new.max() > .5 and new.std() > 0


def test_c5_grid_preserves_contours_above_opencv_noise_area_cutoff():
    features = torch.rand(1, 1, 20, 20, generator=torch.Generator().manual_seed(21))
    features[:, :, :10, :10] = 0
    analyzer = MorphologicalComplexityAnalyzer(device='cpu', metric_backend='cv2')
    old = analyzer.compute_phi_tiles(features, grid_size=8)[1]['contour']
    new = analyzer.compute_phi_tiles(features, grid_size=2)[1]['contour']
    # With 2/3-pixel tiles, no contour can exceed the existing area>10 cutoff.
    # The four 10-pixel tiles preserve finite, distinct measured contours.
    assert torch.count_nonzero(old) == 0
    assert torch.isfinite(new).all() and torch.count_nonzero(new) > 0
    assert new.std() > 0


@pytest.mark.parametrize('kwargs', [dict(grid_sizes=[8, 4]), dict(grid_sizes=[8, 0, 2]),
                                  dict(grid_sizes=[8, 4., 2]), dict(grid_sizes=[8, True, 2]),
                                  dict(grid_size=8, grid_sizes=[8, 4, 2]), dict(grid_size=0)])
def test_invalid_grid_configuration_rejected(kwargs):
    with pytest.raises(ValueError, match='grid'):
        detector(**kwargs)


def test_variable_grid_precompute_roundtrip_and_wrong_replay_shape_rejected():
    model = detector(grid_sizes=[8, 4, 2], bit_mapping='linear', normalize_complexity=True).eval()
    x = torch.rand(1, 3, 96, 64)
    with torch.no_grad():
        model.calibrate([x])
        online, aux = model(x)
        maps = model.precompute_bit_maps(x)
        replay, replay_aux = model(x, precomputed_bit_maps=maps)
    assert [tuple(m.shape[-2:]) for m in maps] == [(8, 8), (4, 4), (2, 2)]
    torch.testing.assert_close(online[0], replay[0])
    for expected, actual in zip(aux['bit_map'], replay_aux['bit_map']):
        torch.testing.assert_close(expected, actual)
    maps[-1] = torch.full((1, 3, 2), 4.)
    with pytest.raises(ValueError, match='grid'):
        model(x, precomputed_bit_maps=maps)


def test_paired_shuffle_preserves_each_variable_grid_budget_and_histogram():
    from mcaq_yolo.core.bit_allocation import tile_areas
    model = detector(grid_sizes=[8, 4, 2], bit_mapping='linear', normalize_complexity=True).eval()
    reference, shuffled = model.paired_shuffle_forward(torch.rand(1, 3, 96, 64))
    for a, b, feat in zip(reference[1]['bit_map'], shuffled[1]['bit_map'], reference[1]['quantized_features']):
        torch.testing.assert_close(a.flatten().sort().values, b.flatten().sort().values)
        areas = tile_areas(a, feat.shape[-2:])
        torch.testing.assert_close((a * areas).sum(), (b * areas).sum())


def test_std_control_uses_level_grids():
    model = detector(allocation_policy='std').eval()
    with torch.no_grad():
        _, aux = model(torch.rand(1, 3, 96, 64))
    assert [tuple(x.shape[-2:]) for x in aux['bit_map']] == [(8, 8), (4, 4), (2, 2)]


@pytest.mark.parametrize('quantization,expected', [({'grid_sizes': [8, 4, 2]}, (8, 4, 2)),
                                                  ({'grid_size': 4}, (4, 4, 4)), ({}, (8, 8, 8))])
def test_inference_reload_preserves_new_and_legacy_grid_policies(tmp_path, quantization, expected):
    from mcaq_yolo.inference import Predictor
    source_kwargs = quantization or {'grid_size': 8}
    model = detector(**source_kwargs).eval()
    checkpoint = tmp_path / 'best.pt'
    torch.save({'model_state_dict': model.state_dict()}, checkpoint)
    cfg = {'model': {'name': 'yolov8n.yaml'}, 'quantization': quantization}
    config_path = tmp_path / 'resolved_config.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    loader = Predictor.__new__(Predictor)
    loader.device = torch.device('cpu')
    restored = loader._load_model(str(checkpoint), None).eval()
    assert restored.feature_grid_sizes == expected
    assert set(restored.state_dict()) == set(model.state_dict())
    x = torch.rand(1, 3, 96, 64)
    with torch.no_grad():
        a = model(x)[1]['bit_map']
        b = restored(x)[1]['bit_map']
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)


def test_manifest_records_feature_grids_separately_from_input_scoring(tmp_path):
    from mcaq_yolo.experiments import write_manifest
    model = detector(grid_sizes=[8, 4, 2])
    write_manifest({}, tmp_path, model)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['model']['feature_grid_sizes'] == [8, 4, 2]
    assert manifest['model']['sample_score_grid_size'] == 8


def _tiny_training_config(tmp_path):
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    weights = tmp_path / 'initial.pt'
    YOLO('yolov8n.yaml').save(weights)
    for split in ('train', 'val'):
        (tmp_path / 'images' / split).mkdir(parents=True)
        (tmp_path / 'labels' / split).mkdir(parents=True)
        for i in range(2):
            pixels = np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(tmp_path / 'images' / split / f'{i}.png')
            (tmp_path / 'labels' / split / f'{i}.txt').write_text('0 .5 .5 .3 .3\n')
    data = tmp_path / 'data.yaml'
    data.write_text(yaml.safe_dump(dict(path=str(tmp_path), train='images/train',
                                       val='images/val', names={0: 'object'})))
    return dict(seed=3, epochs=1, batch_size=2, device='cpu', output_dir=str(tmp_path / 'run'),
                  model=dict(name=str(weights), teacher_path=str(weights)),
                  data=dict(yaml_path=str(data), imgsz=64, num_workers=0),
                  curriculum=dict(sample_enabled=False, precision_enabled=False),
                  scheduler=dict(warmup_epochs=0), training=dict(amp=False),
                  distillation=dict(enabled=False))


def test_trainer_persists_default_grids_for_prediction_reload(tmp_path):
    from mcaq_yolo.inference import Predictor
    from mcaq_yolo.train import Trainer
    config = _tiny_training_config(tmp_path)
    trainer = Trainer(config)
    trainer.train()
    saved = yaml.safe_load((tmp_path / 'run' / 'resolved_config.yaml').read_text())
    assert saved['quantization']['grid_sizes'] == [8, 4, 2]
    assert 'grid_size' not in saved['quantization']
    loader = Predictor.__new__(Predictor)
    loader.device = torch.device('cpu')
    restored = loader._load_model(str(tmp_path / 'run' / 'best.pt'), None)
    assert restored.feature_grid_sizes == (8, 4, 2)


@pytest.mark.parametrize('grid_config,expected', [({}, [8, 8, 8]),
                                                ({'grid_sizes': [8, 4, 2]}, [8, 4, 2])])
def test_paired_cli_preserves_checkpoint_grid_when_old_config_omits_it(tmp_path, grid_config, expected):
    from mcaq_yolo.experiments import main
    config = _tiny_training_config(tmp_path)
    config['quantization'] = grid_config
    config_path = tmp_path / 'archived_config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    source = detector(**(grid_config or {'grid_size': 8}))
    checkpoint = tmp_path / 'adaptive.pt'
    torch.save(source.state_dict(), checkpoint)
    output = tmp_path / 'paired'
    main(['--config', str(config_path), '--output-dir', str(output),
          '--paired-checkpoint', str(checkpoint)])
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['model']['feature_grid_sizes'] == expected
    assert (output / 'paired_result.json').is_file()
