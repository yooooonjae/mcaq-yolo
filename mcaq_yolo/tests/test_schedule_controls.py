"""Regression coverage for feasible precision schedules and matched controls."""
import pytest
import torch

from mcaq_yolo.core.curriculum import CurriculumScheduler


def test_default_budget_reaches_endpoint_when_transition_finishes():
    schedule = CurriculumScheduler()
    assert schedule.get_target_bits(20) == 8.0
    assert schedule.get_target_bits(50) == 4.0
    assert schedule.get_target_bits(300) == 4.0


def test_first_quantized_epoch_has_diversity_and_a_reachable_budget():
    from mcaq_yolo.core.bit_allocation import interpolate_precision
    schedule = CurriculumScheduler()
    base = torch.tensor([2., 4., 6., 8.], requires_grad=True)
    bits = interpolate_precision(base, schedule.get_precision_progress(21), 8.)
    torch.testing.assert_close(bits, torch.tensor([7.8, 7.8666667, 7.9333333, 8.]))
    bits.sum().backward()
    torch.testing.assert_close(base.grad, torch.full((4,), 1 / 30))
    for epoch in range(301):
        r = schedule.get_precision_progress(epoch)
        endpoints = interpolate_precision(torch.tensor([2., 8.]), r, 8.)
        assert endpoints[0] - 1e-6 <= schedule.get_target_bits(epoch) <= endpoints[1] + 1e-6


@pytest.mark.parametrize('sample,precision', [(False, False), (True, False), (False, True), (True, True)])
def test_curricula_can_be_disabled_independently(sample, precision):
    schedule = CurriculumScheduler(sample_enabled=sample, precision_enabled=precision)
    assert (schedule.get_complexity_threshold(1) < 1) == sample
    assert (schedule.get_stage(1) == 1) == precision
    assert schedule.get_precision_progress(1) == (0. if precision else 1.)
    assert schedule.get_target_bits(1) == (8. if precision else 4.)


def test_shuffling_unequal_tiles_preserves_histogram_and_element_budget():
    from mcaq_yolo.core.bit_allocation import apply_allocation_policy, tile_areas
    bits = torch.linspace(2, 8, 64).reshape(1, 8, 8).requires_grad_()
    shuffled = apply_allocation_policy(bits, (20, 20), 'shuffle', seed=17)
    assert not torch.equal(bits, shuffled)
    torch.testing.assert_close(bits.flatten().sort().values, shuffled.flatten().sort().values)
    areas = tile_areas(bits, (20, 20))
    torch.testing.assert_close((bits * areas).sum(), (shuffled * areas).sum())
    shuffled.sum().backward()
    torch.testing.assert_close(bits.grad, torch.ones_like(bits))
    torch.testing.assert_close(shuffled, apply_allocation_policy(bits, (20, 20), 'shuffle', seed=17))


def test_uniform_control_is_spatially_constant_and_layerwise_preserves_element_cost():
    from mcaq_yolo.core.bit_allocation import apply_allocation_policy, tile_areas
    bits = torch.linspace(2, 8, 64).reshape(1, 8, 8)
    uniform = apply_allocation_policy(bits, (20, 20), 'uniform', uniform_bits=4)
    assert torch.all(uniform == 4)
    layer = apply_allocation_policy(bits, (20, 20), 'layerwise')
    assert layer.std() < 1e-6
    areas = tile_areas(bits, (20, 20))
    torch.testing.assert_close((bits * areas).sum(), (layer * areas).sum())


def test_actual_detector_applies_progress_and_gate_control():
    from mcaq_yolo.models.mcaq_yolo import MCAQYOLO
    model = MCAQYOLO(model_name='yolov8n.yaml', pretrained=False, device='cpu',
                     allocation_policy='uniform', smooth_transitions=False).train()
    x = torch.rand(2, 3, 64, 64)
    _, warm = model(x, precision_progress=0, quantize=False)
    _, transition = model(x, precision_progress=0.5)
    _, full = model(x, precision_progress=1)
    for aux, expected in [(warm, 8.), (transition, 6.), (full, 4.)]:
        assert all(torch.all(b == expected) for b in aux['bit_map'])
    assert all(not q.smooth_transitions for q in model.quantizers.values())


@pytest.mark.parametrize('kwargs', [dict(transition_epochs=20), dict(target_bits=9), dict(min_bits=5)])
def test_invalid_precision_bounds_rejected(kwargs):
    with pytest.raises(ValueError):
        CurriculumScheduler(**kwargs)


def test_diagnostics_do_not_confuse_tile_mean_with_element_mean():
    from mcaq_yolo.utils.bit_diagnostics import BitDiagnostics
    stats = BitDiagnostics()
    # The two tiles cover 2 and 1 columns respectively, with three channels.
    stats.update({'bit_map': [torch.tensor([[[2., 8.]]])],
                  'feature_layers': [4], 'quantized_features': [torch.zeros(1, 3, 1, 3)]})
    result = stats.result(target=4.)
    assert result['policy_mean_bits'] == 5.
    assert result['element_mean_bits'] == 4.
    assert result['target_gap'] == 1.
    assert result['levels']['4']['saturated_max_fraction'] == .5
    assert result['levels']['4']['histogram']['2'] == 1


def test_descriptor_diagnostics_report_variance_and_degenerate_values():
    from mcaq_yolo.utils.bit_diagnostics import BitDiagnostics
    stats = BitDiagnostics()
    stats.update({'bit_map': [torch.tensor([[[2., 8.]]])], 'feature_layers': [4],
                  'quantized_features': [torch.zeros(1, 3, 1, 3)],
                  'complexity_map': [torch.tensor([[[0., 1.]]])],
                  'detailed_metrics': [{'fractal': torch.tensor([[[.5, .5]]]),
                                       'contour': torch.tensor([[[0., 1.]]])}]})
    result = stats.result()['levels']['4']
    assert result['descriptors']['fractal']['std'] == 0.
    assert result['descriptors']['fractal']['fallback_value_fraction'] == 1.
    assert result['descriptors']['contour']['zero_fraction'] == .5
    assert result['complexity']['std'] == .5
    assert result['qdq_tensor_bytes_per_image'] == 36.
    assert result['ideal_bit_payload_bytes_per_image'] == 4.5


def test_suite_changes_only_named_factors_and_does_not_mutate_base():
    from copy import deepcopy
    from mcaq_yolo.experiments import make_variant
    base = {'model': {'name': 'yolov8n'}, 'distillation': {'enabled': True},
            'quantization': {'target_bits': 4}, 'curriculum': {'warmup_epochs': 20}}
    original = deepcopy(base)
    both = make_variant(base, 'both', 7, 'out')
    neither = make_variant(base, 'neither', 7, 'out')
    assert both['distillation'] == neither['distillation'] == {'enabled': True}
    assert neither['curriculum']['sample_enabled'] is False
    assert neither['curriculum']['precision_enabled'] is False
    assert both['curriculum']['sample_enabled'] is True
    assert both['curriculum']['precision_enabled'] is True
    uniform = make_variant(base, 'uniform', 7, 'out')
    assert uniform['quantization']['smooth_transitions'] == both['quantization']['smooth_transitions']
    assert uniform['quantization']['allocation_policy'] == 'uniform'
    assert make_variant(base, 'gate_off', 7, 'out')['quantization']['smooth_transitions'] is False
    assert base == original


def test_real_trainer_crosses_schedule_boundary_and_saves_control_evidence(tmp_path):
    import json
    import numpy as np
    import yaml
    from PIL import Image
    from ultralytics import YOLO
    from mcaq_yolo.train import Trainer
    weights = tmp_path / 'initial.pt'
    YOLO('yolov8n.yaml').save(weights)
    for split in ('train', 'val'):
        (tmp_path / 'images' / split).mkdir(parents=True)
        (tmp_path / 'labels' / split).mkdir(parents=True)
        for i in range(2):
            rng = np.random.default_rng(i)
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(tmp_path / 'images' / split / f'{i}.png')
            (tmp_path / 'labels' / split / f'{i}.txt').write_text('0 0.5 0.5 0.3 0.3\n')
    data = tmp_path / 'data.yaml'
    data.write_text(yaml.safe_dump(dict(path=str(tmp_path), train='images/train', val='images/val', names={0: 'object'})))
    config = dict(seed=3, epochs=3, batch_size=2, device='cpu', output_dir=str(tmp_path / 'run'),
                  model=dict(name=str(weights), teacher_path=str(weights)),
                  data=dict(yaml_path=str(data), imgsz=64, num_workers=0),
                  curriculum=dict(warmup_epochs=1, transition_epochs=2),
                  scheduler=dict(warmup_epochs=0), quantization=dict(allocation_policy='uniform'),
                  training=dict(amp=False))
    trainer = Trainer(config)
    assert trainer.train_dataset.imgsz == 64
    trainer.train()
    history = [json.loads(line) for line in (tmp_path / 'run/history.jsonl').read_text().splitlines()]
    assert [row['train']['bit_diagnostics']['policy_mean_bits'] for row in history] == [8., 4., 4.]
    assert [row['precision_progress'] for row in history] == [0., 1., 1.]
    result = json.loads((tmp_path / 'run/result.json').read_text())
    assert result['final_policy_trained'] is True
    assert result['evaluation']['bit_diagnostics']['policy_mean_bits'] == 4.
    assert (tmp_path / 'run/manifest.json').is_file()


def test_new_ultralytics_raw_dict_still_contributes_logit_distillation():
    from mcaq_yolo.models.mcaq_yolo import kd_logit_loss
    boxes = torch.ones(2, 64, 20, requires_grad=True)
    scores = torch.ones(2, 6, 20, requires_grad=True)
    loss = kd_logit_loss({'boxes': boxes, 'scores': scores},
                         (torch.zeros(2, 10, 20), {'boxes': boxes.detach() * 0, 'scores': scores.detach() * 0}))
    assert loss is not None
    loss.backward()
    assert boxes.grad.abs().sum() > 0 and scores.grad.abs().sum() > 0


def test_simple_std_allocator_prefers_variable_tiles():
    from mcaq_yolo.core.bit_allocation import feature_std_complexity
    x = torch.tensor([0., 0., 0., 2.]).reshape(1, 1, 1, 4)
    torch.testing.assert_close(feature_std_complexity(x, 2), torch.tensor([[[0., 1.]]]))


@pytest.mark.parametrize('policy,gate', [('uniform', True), ('shuffle', False)])
def test_inference_reload_keeps_saved_control_policy(tmp_path, policy, gate):
    import yaml
    from mcaq_yolo.models.mcaq_yolo import MCAQYOLO
    from mcaq_yolo.inference import Predictor
    model = MCAQYOLO(model_name='yolov8n.yaml', pretrained=False, device='cpu',
                     allocation_policy=policy, smooth_transitions=gate, control_seed=17).eval()
    weights = tmp_path / 'best.pt'
    torch.save(model.state_dict(), weights)
    config = dict(seed=17, model={'name': 'yolov8n.yaml'},
                  quantization={'allocation_policy': policy, 'smooth_transitions': gate})
    (tmp_path / 'resolved_config.yaml').write_text(yaml.safe_dump(config))
    predictor = Predictor.__new__(Predictor)
    predictor.device = torch.device('cpu')
    restored = predictor._load_model(str(weights), None).eval()
    assert restored.allocation_policy == policy
    assert restored.control_seed == 17
    assert all(q.smooth_transitions == gate for q in restored.quantizers.values())
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0][0], restored(x)[0][0])


def test_paired_shuffle_preserves_all_levels_of_the_same_forward():
    from mcaq_yolo.models.mcaq_yolo import MCAQYOLO
    from mcaq_yolo.core.bit_allocation import tile_areas
    model = MCAQYOLO(model_name='yolov8n.yaml', pretrained=False, device='cpu',
                     bit_mapping='linear', normalize_complexity=True).eval()
    with torch.no_grad():
        (_, ref), (_, shuffled) = model.paired_shuffle_forward(torch.rand(2, 3, 96, 96))
    assert shuffled['complexity_map'] == [None] * 3
    changed = False
    for a, b, features in zip(ref['bit_map'], shuffled['bit_map'], ref['quantized_features']):
        changed |= not torch.equal(a, b)
        torch.testing.assert_close(a.flatten(1).sort().values, b.flatten(1).sort().values)
        areas = tile_areas(a, features.shape[-2:])
        torch.testing.assert_close((a * areas).sum(), (b * areas).sum())
    assert changed
