"""Matched training controls: python -m mcaq_yolo.experiments --help."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics
import yaml


VARIANTS = ('both', 'neither', 'sample_only', 'precision_only', 'uniform',
            'shuffle', 'gate_off', 'layerwise', 'std', 'legacy')


def make_variant(base, variant, seed, output_dir):
    if variant not in VARIANTS:
        raise ValueError(f'unknown variant: {variant}')
    config = copy.deepcopy(base)
    config.update(seed=int(seed), output_dir=str(output_dir), experiment_variant=variant)
    c = config.setdefault('curriculum', {})
    c.update(enabled=True, schedule_mode='legacy_multiply' if variant == 'legacy' else 'interpolate',
             sample_enabled=variant not in ('neither', 'precision_only'),
             precision_enabled=variant not in ('neither', 'sample_only'))
    q = config.setdefault('quantization', {})
    q.update(allocation_policy=variant if variant in ('uniform', 'shuffle', 'layerwise', 'std') else 'adaptive',
             smooth_transitions=variant != 'gate_off')
    return config


def file_hash(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(config, output_dir, model=None, datasets=None):
    import torch
    import ultralytics
    root = Path(__file__).resolve().parent
    manifest = dict(config=config, python=platform.python_version(), torch=torch.__version__,
                    ultralytics=ultralytics.__version__, cuda=torch.version.cuda,
                    source_sha256={p.relative_to(root).as_posix(): file_hash(p)
                                   for p in sorted(root.rglob('*.py'))},
                    input_sha256={})
    for key, value in [('data_yaml', config.get('data', {}).get('yaml_path')),
                       ('teacher', config.get('model', {}).get('teacher_path')),
                       ('student', config.get('model', {}).get('name'))]:
        if value:
            manifest['input_sha256'][key] = file_hash(value)
    if model is not None:
        manifest['model'] = dict(architecture=model.model.yaml,
                                 classes=model.model.names,
                                 feature_layers=model.backbone_out_indices,
                                 allocation_policy=model.allocation_policy,
                                 gate=model.smooth_transitions,
                                 descriptor_backend=model.complexity_analyzer.metric_backend)
    if datasets:
        index = {}
        for split, dataset in datasets.items():
            index[split] = [dict(image=str(Path(image).resolve()), label=str(Path(label).resolve()),
                                 label_sha256=file_hash(label))
                            for image, label in zip(dataset.im_files, dataset.label_files)]
        index_path = Path(output_dir, 'dataset_index.json')
        index_path.write_text(json.dumps(index, indent=2), encoding='utf-8')
        manifest['dataset_index_sha256'] = file_hash(index_path)
    Path(output_dir, 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')


def collect_results(root):
    rows = []
    for path in sorted(Path(root).glob('*/seed_*/result.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        manifest = json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
        cfg = manifest['config']
        evaluation = result['evaluation']
        bits = evaluation.get('bit_diagnostics', {})
        effort = result.get('training_effort', {})
        rows.append(dict(variant=cfg['experiment_variant'], seed=cfg['seed'],
                         mAP50=evaluation['mAP50'], mAP50_95=evaluation.get('mAP50_95'),
                         AP_small=evaluation.get('AP_small'), AP_medium=evaluation.get('AP_medium'),
                         AP_large=evaluation.get('AP_large'),
                         selection_split=result.get('selection_split', 'val'),
                         evaluation_split=result.get('evaluation_split', 'val'),
                         evaluation_scope=result.get('evaluation_scope', 'validation_only'),
                         evaluation_protocol=evaluation.get('protocol', 'reference_voc'),
                         protocol_details=json.dumps(evaluation.get('protocol_details', {}), sort_keys=True),
                         processed_samples=effort.get('processed_samples'),
                         optimizer_updates=effort.get('optimizer_updates'),
                         training_wall_seconds=effort.get('training_wall_seconds'),
                         policy_mean_bits=bits.get('policy_mean_bits'),
                         element_mean_bits=bits.get('element_mean_bits'),
                         target_gap=bits.get('target_gap'),
                         final_policy_trained=result['final_policy_trained'],
                         schedule_completed=result['schedule_completed']))
    if rows:
        with Path(root, 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        aggregates = {}
        # Keep validation-only results and distinct metric definitions separate.
        group_fields = ('variant', 'evaluation_split', 'evaluation_scope', 'evaluation_protocol', 'protocol_details')
        groups = sorted({tuple(row[key] for key in group_fields) for row in rows})
        metric_fields = ('mAP50', 'mAP50_95', 'AP_small', 'AP_medium', 'AP_large',
                         'processed_samples', 'optimizer_updates', 'training_wall_seconds')
        for group in groups:
            members = [row for row in rows if tuple(row[key] for key in group_fields) == group
                       and row['schedule_completed']]
            if not members:
                continue
            aggregate = dict(zip(group_fields, group))
            aggregate['protocol_details'] = json.loads(aggregate['protocol_details'])
            aggregate['n'] = len(members)
            for metric in metric_fields:
                values = [row[metric] for row in members if row[metric] is not None]
                aggregate[f'n_{metric}'] = len(values)
                aggregate[f'mean_{metric}'] = statistics.mean(values) if values else None
                aggregate[f'std_{metric}'] = statistics.stdev(values) if len(values) > 1 else None
            key = '/'.join(group[:4]) + '/' + hashlib.sha256(group[4].encode()).hexdigest()[:8]
            aggregates[key] = aggregate
        Path(root, 'aggregate.json').write_text(json.dumps(aggregates, indent=2), encoding='utf-8')
    return rows


def plot_histories(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths = []
    for history in sorted(Path(root).glob('*/seed_*/history.jsonl')):
        rows = [json.loads(line) for line in history.read_text(encoding='utf-8').splitlines()]
        if not rows:
            continue
        epochs = [r['epoch'] for r in rows]
        stats = [r['train']['bit_diagnostics'] for r in rows]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for level in stats[-1]['levels']:
            level_stats = [s['levels'][level] for s in stats]
            axes[0, 0].plot(epochs, [s['mean_bits'] for s in level_stats], label=f'layer {level}')
            axes[0, 0].fill_between(epochs, [s['min_bits'] for s in level_stats],
                                    [s['max_bits'] for s in level_stats], alpha=.12)
            axes[0, 1].plot(epochs, [s['saturated_max_fraction'] for s in level_stats], label=f'layer {level}')
        axes[0, 0].set_title('Continuous training bits (band: min/max)')
        axes[0, 1].set_title('Fraction at maximum precision')
        axes[1, 0].plot(epochs, [r['target_bits'] for r in rows], label='target')
        axes[1, 0].plot(epochs, [s['policy_mean_bits'] for s in stats], label='policy mean')
        axes[1, 0].plot(epochs, [s['element_mean_bits'] for s in stats], label='element mean')
        axes[1, 0].set_title('Target and achieved bit budgets')
        result_path = history.with_name('result.json')
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding='utf-8'))
            distributions = [('last training (rounded)', stats[-1]),
                             ('selected endpoint inference', result['evaluation']['bit_diagnostics'])]
            for index, (label, data) in enumerate(distributions):
                counts = [sum(s['histogram'][str(b)] for s in data['levels'].values()) for b in range(2, 9)]
                axes[1, 1].bar([b + (index - .5) * .35 for b in range(2, 9)],
                                [n / max(1, sum(counts)) for n in counts], width=.35, label=label)
        axes[1, 1].set_title('Tile-bit histogram')
        for ax in axes.flat:
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.set_xlabel('epoch')
        axes[1, 1].set_xlabel('integer bits')
        fig.suptitle(f'{history.parent.parent.name} / {history.parent.name}')
        output = history.with_name('bit_diagnostics.png')
        fig.savefig(output, dpi=160)
        plt.close(fig)
        paths.append(str(output))
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/train_config.yaml')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data', help='YOLO dataset YAML; its train/val/test entries take precedence')
    parser.add_argument('--student', help='local student .pt or YOLO model name')
    parser.add_argument('--teacher', help='local teacher .pt')
    parser.add_argument('--device')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--plot-only', action='store_true', help='render bit diagnostics from saved histories')
    parser.add_argument('--paired-checkpoint', help='evaluate fixed-map shuffle on an adaptive checkpoint; no training')
    args = parser.parse_args(argv)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        for path in plot_histories(root):
            print(path)
        return
    if args.collect_only:
        collect_results(root)
        return
    base = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.data:
        base.setdefault('data', {})['yaml_path'] = str(Path(args.data).resolve())
        for key in ('train', 'val', 'test'):
            base['data'].pop(key, None)
    for key, value in [('name', args.student), ('teacher_path', args.teacher)]:
        if value:
            base.setdefault('model', {})[key] = str(Path(value).resolve()) if Path(value).exists() else value
    if args.device:
        base['device'] = args.device
    if args.paired_checkpoint:
        import torch
        from .train import Trainer
        if base.get('quantization', {}).get('allocation_policy', 'adaptive') != 'adaptive':
            raise ValueError('paired-checkpoint requires an adaptive run config')
        if (root / 'paired_result.json').exists():
            raise FileExistsError('paired_result.json already exists; use a fresh output directory')
        base['output_dir'] = str(root)
        trainer = Trainer(base)
        trainer.model.load_state_dict(torch.load(args.paired_checkpoint, map_location=trainer.device, weights_only=True))
        datasets = {'train': trainer.train_dataset, 'val': trainer.val_dataset}
        if trainer.test_dataset is not None:
            datasets['test'] = trainer.test_dataset
        write_manifest(trainer.config, root, trainer.model, datasets)
        split = 'test' if trainer.test_loader is not None else 'val'
        reference = trainer.evaluate(precision_progress=1., split=split)
        shuffled = trainer.evaluate(precision_progress=1., paired_shuffle=True, split=split)
        delta = (shuffled['mAP50'] - reference['mAP50']
                 if shuffled['mAP50'] is not None and reference['mAP50'] is not None else None)
        result = dict(checkpoint_sha256=file_hash(args.paired_checkpoint), reference=reference,
                      shuffled=shuffled, delta_mAP50=delta, evaluation_split=split,
                      evaluation_scope='held_out_test' if split == 'test' else 'validation_only')
        (root / 'paired_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return
    for variant in args.variants:
        for seed in args.seeds:
            run = root / variant / f'seed_{seed}'
            if (run / 'history.jsonl').exists():
                raise FileExistsError(f'{run} already contains a run; use a fresh directory')
            run.mkdir(parents=True, exist_ok=True)
            config = make_variant(base, variant, seed, run)
            (run / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            print(f'{variant}, seed={seed}: {run}')
            if not args.prepare_only:
                from .train import Trainer
                Trainer(config).train()
                collect_results(root)


if __name__ == '__main__':
    main()
