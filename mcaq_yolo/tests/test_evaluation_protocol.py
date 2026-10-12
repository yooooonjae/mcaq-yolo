"""Held-out evaluation, real work accounting, and protocol separation."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image


@pytest.fixture
def dataset_yaml(tmp_path):
    for split, count in [('train', 3), ('val', 2), ('test', 1)]:
        (tmp_path / 'images' / split).mkdir(parents=True)
        (tmp_path / 'labels' / split).mkdir(parents=True)
        for index in range(count):
            pixels = np.random.default_rng(index).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(tmp_path / 'images' / split / f'{index}.png')
            (tmp_path / 'labels' / split / f'{index}.txt').write_text('0 0.5 0.5 0.3 0.3\n')
    path = tmp_path / 'data.yaml'
    path.write_text(yaml.safe_dump(dict(path=str(tmp_path), train='images/train',
                                      val='images/val', test='images/test', names={0: 'object'})))
    return path


def _dataset_trainer(data):
    from mcaq_yolo.train import Trainer
    trainer = Trainer.__new__(Trainer)
    trainer.data_cfg = data
    trainer.imgsz, trainer.batch_size, trainer.num_workers = 64, 2, 0
    trainer.device = torch.device('cpu')
    trainer.teacher_model = SimpleNamespace()
    return trainer


def test_dataset_yaml_test_split_is_loaded_and_override_can_disable_it(dataset_yaml):
    trainer = _dataset_trainer(dict(yaml_path=str(dataset_yaml)))
    trainer._init_datasets()
    assert trainer.test_dataset is not None and len(trainer.test_dataset) == 1
    disabled = _dataset_trainer(dict(yaml_path=str(dataset_yaml), test=None))
    disabled._init_datasets()
    assert disabled.test_dataset is None


def test_explicit_test_override_rejects_validation_image_overlap(dataset_yaml):
    trainer = _dataset_trainer(dict(yaml_path=str(dataset_yaml), test='images/val'))
    with pytest.raises(ValueError, match='overlap'):
        trainer._init_datasets()


@pytest.mark.parametrize('held_out,protocol', [(True, 'reference_voc'), (False, 'reference_voc'),
                                               (True, 'coco_bbox')])
def test_real_training_evaluates_test_only_after_selection_and_counts_partial_batches(
        tmp_path, dataset_yaml, monkeypatch, held_out, protocol):
    if protocol == 'coco_bbox':
        pytest.importorskip('pycocotools')
    from ultralytics import YOLO
    from mcaq_yolo.train import Trainer
    weights = tmp_path / 'initial.pt'
    YOLO('yolov8n.yaml').save(weights)
    data = dict(yaml_path=str(dataset_yaml), imgsz=64, num_workers=0)
    if not held_out:
        data['test'] = None
    config = dict(seed=5, epochs=2, batch_size=2, device='cpu', output_dir=str(tmp_path / 'run'),
                  model=dict(name=str(weights), teacher_path=str(weights)), data=data,
                  curriculum=dict(warmup_epochs=0, transition_epochs=1, sample_enabled=False,
                                  precision_enabled=False),
                  scheduler=dict(warmup_epochs=0), quantization=dict(allocation_policy='uniform'),
                  distillation=dict(enabled=False), training=dict(amp=False),
                  evaluation=dict(protocol=protocol))
    trainer = Trainer(config)
    calls = []
    evaluate = trainer.evaluate
    def observe_evaluation(*args, **kwargs):
        split = kwargs.get('split', 'val')
        calls.append(split)
        if split == 'test':
            assert (trainer.output_dir / 'best.pt').is_file()
            assert len((trainer.output_dir / 'history.jsonl').read_text().splitlines()) == 2
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(trainer, 'evaluate', observe_evaluation)
    trainer.train()
    assert calls == ['val', 'val', 'test' if held_out else 'val']
    history = [json.loads(line) for line in (trainer.output_dir / 'history.jsonl').read_text().splitlines()]
    for row in history:
        assert row['train']['processed_samples'] == 3
        assert row['train']['optimizer_updates'] == 2
        assert row['train']['training_wall_seconds'] > 0
    assert [row['training_effort']['processed_samples'] for row in history] == [3, 6]
    result = json.loads((trainer.output_dir / 'result.json').read_text())
    assert result['selection_split'] == 'val'
    assert result['evaluation_split'] == ('test' if held_out else 'val')
    assert result['evaluation_scope'] == ('held_out_test' if held_out else 'validation_only')
    assert result['evaluation']['protocol'] == protocol
    assert result['evaluation']['mAP50_95'] is not None
    if protocol == 'reference_voc':
        assert result['evaluation']['AP_small'] is None
    else:
        assert result['evaluation']['AP_small'] is not None
    assert result['training_effort']['processed_samples'] == 6
    assert result['training_effort']['optimizer_updates'] == 4
    assert result['training_effort']['training_wall_seconds'] == pytest.approx(
        sum(row['train']['training_wall_seconds'] for row in history))
    index = json.loads((trainer.output_dir / 'dataset_index.json').read_text())
    assert ('test' in index) is held_out


def test_collector_does_not_pool_validation_test_or_metric_protocols(tmp_path):
    from mcaq_yolo.experiments import collect_results
    settings = [('val', 'reference_voc', .1), ('test', 'reference_voc', .4), ('test', 'coco_bbox', .9)]
    for seed, (split, protocol, score) in enumerate(settings):
        run = tmp_path / 'both' / f'seed_{seed}'
        run.mkdir(parents=True)
        (run / 'manifest.json').write_text(json.dumps({'config': {'seed': seed, 'experiment_variant': 'both'}}))
        result = dict(final_policy_trained=True, schedule_completed=True, selection_split='val',
                      evaluation_split=split, evaluation_scope='held_out_test' if split == 'test' else 'validation_only',
                      training_effort=dict(processed_samples=9, optimizer_updates=6, training_wall_seconds=2.5),
                      evaluation=dict(mAP50=score, mAP50_95=score / 2, AP_small=None, AP_medium=None,
                                      AP_large=None, protocol=protocol,
                                      protocol_details={'annotation_scope': 'ordinary_bbox_only'}))
        (run / 'result.json').write_text(json.dumps(result))
    rows = collect_results(tmp_path)
    assert {(r['evaluation_split'], r['evaluation_protocol']) for r in rows} == {
        ('val', 'reference_voc'), ('test', 'reference_voc'), ('test', 'coco_bbox')}
    assert all(row['processed_samples'] == 9 and row['optimizer_updates'] == 6 for row in rows)
    assert all(row['training_wall_seconds'] == 2.5 for row in rows)
    assert all(row['mAP50_95'] is not None and row['protocol_details'] for row in rows)
    aggregate = json.loads((tmp_path / 'aggregate.json').read_text())
    assert len(aggregate) == 3
    assert sorted(group['mean_mAP50'] for group in aggregate.values()) == [.1, .4, .9]


def _perfect_box_trainer():
    from mcaq_yolo.train import Trainer
    class PerfectBox(torch.nn.Module):
        max_bits, target_bits, bit_mapper = 8, 4, None
        def forward(self, images, **kwargs):
            return torch.tensor([[[32.], [32.], [16.], [16.], [.9]]]), {
                'feature_layers': [], 'bit_map': [], 'quantized_features': []}
        def loss_fn(self, *args, **kwargs):
            return torch.tensor(0.), {'loss_det': torch.tensor(0.)}
    trainer = Trainer.__new__(Trainer)
    trainer.config = {'evaluation': {'protocol': 'coco_bbox'}}
    trainer.model, trainer.device, trainer.use_amp = PerfectBox(), torch.device('cpu'), False
    trainer.val_loader = [dict(img=torch.zeros(1, 3, 64, 64), cls=torch.tensor([[0.]]),
                               batch_idx=torch.tensor([0]), bboxes=torch.tensor([[.5, .5, .25, .25]]),
                               ori_shape=[(128, 256)], ratio_pad=[((.25, .25), (0., 16.))])]
    return trainer


def test_coco_size_ap_uses_original_pixels_after_letterbox_inverse():
    pytest.importorskip('pycocotools')
    metrics = _perfect_box_trainer().evaluate()
    # 16x16 network pixels become 64x64 original pixels: medium, not small.
    assert metrics['protocol'] == 'coco_bbox'
    assert metrics['AP_medium'] == pytest.approx(1.)
    assert metrics['AP_small'] is None
    assert metrics['AP_large'] is None
    assert metrics['protocol_details']['area_coordinates'] == 'original_image_pixels'


def test_coco_evaluation_rejects_missing_original_image_geometry():
    trainer = _perfect_box_trainer()
    del trainer.val_loader[0]['ratio_pad']
    with pytest.raises(ValueError, match='ori_shape.*ratio_pad'):
        trainer.evaluate()


def test_amp_overflow_skips_do_not_count_as_optimizer_updates(monkeypatch):
    from contextlib import nullcontext
    import mcaq_yolo.train as training
    from mcaq_yolo.core.curriculum import CurriculumScheduler
    class TinyModel(torch.nn.Module):
        max_bits = 8
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.bit_mapper = SimpleNamespace(enforce_weight_constraints=lambda: None)
        def forward(self, images, *args, **kwargs):
            return self.weight, {'feature_layers': [], 'bit_map': [], 'quantized_features': []}
        def loss_fn(self, outputs, batch, *args, **kwargs):
            loss = outputs * batch['multiplier']
            return loss, {'loss_det': loss}
    trainer = training.Trainer.__new__(training.Trainer)
    trainer.model, trainer.teacher_module = TinyModel(), torch.nn.Identity()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.1)
    trainer.scaler = torch.amp.GradScaler('cpu', init_scale=16.)
    trainer.device, trainer.use_amp, trainer.epochs = torch.device('cpu'), True, 1
    trainer.distill_cfg = {'enabled': False}
    trainer.curriculum_cfg = {}
    trainer.curriculum = CurriculumScheduler(warmup_epochs=0, transition_epochs=1, precision_enabled=False)
    trainer._build_curriculum_loader = lambda _: [dict(img=torch.ones(2, 3, 2, 2), multiplier=float('inf')),
                                                 dict(img=torch.ones(1, 3, 2, 2), multiplier=1.)]
    # The trainer's CUDA autocast context is irrelevant to testing a real CPU scaler.
    monkeypatch.setattr(training, 'autocast', lambda *args, **kwargs: nullcontext())
    metrics = trainer.train_epoch(1)
    assert metrics['processed_samples'] == 3
    assert metrics['optimizer_updates'] == 1
    assert trainer.model.weight.item() == pytest.approx(.9)


def test_run_augmentation_overrides_teacher_and_records_effective_settings(dataset_yaml):
    trainer = _dataset_trainer(dict(yaml_path=str(dataset_yaml), test=None))
    trainer.config = {'augmentation': {'mosaic': .25}}
    trainer.teacher_model.overrides = dict(mosaic=.8, mixup=.4, degrees=7.)
    trainer._init_datasets()
    assert trainer.config['resolved_augmentation']['mosaic'] == .25
    assert trainer.config['resolved_augmentation']['mixup'] == 0.
    assert trainer.config['resolved_augmentation']['degrees'] == 7.
