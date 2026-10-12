"""Detached streaming diagnostics for actual maps consumed by quantization."""
import math
import torch
from ..core.bit_allocation import tile_areas


class BitDiagnostics:
    def __init__(self, max_bits=8):
        self.max_bits = max_bits
        self.levels = {}

    @torch.no_grad()
    def update(self, aux):
        for index, (layer, bits, features) in enumerate(zip(aux['feature_layers'], aux['bit_map'], aux['quantized_features'])):
            b = bits.detach().float()
            areas = tile_areas(b, features.shape[-2:]) * features.shape[1]
            record = self.levels.setdefault(str(layer), dict(
                tiles=0, sum=0., min=8., max=2., saturated=0, element_sum=0., elements=0.,
                histogram={str(k): 0 for k in range(2, 9)}, images=0,
                qdq_bytes=0, map_bytes=0, measurements={}))
            record['images'] += b.shape[0]
            record['qdq_bytes'] += features.numel() * features.element_size()
            record['map_bytes'] += bits.numel() * bits.element_size()
            record['tiles'] += b.numel()
            record['sum'] += b.sum().item()
            record['min'] = min(record['min'], b.min().item())
            record['max'] = max(record['max'], b.max().item())
            record['saturated'] += (b >= self.max_bits - 1e-6).sum().item()
            record['element_sum'] += (b * areas).sum().item()
            record['elements'] += areas.sum().item() * b.shape[0]
            for k in record['histogram']:
                record['histogram'][k] += (b.round() == int(k)).sum().item()
            details = aux.get('detailed_metrics', [])
            values = dict(details[index]) if isinstance(details, list) and index < len(details) else {}
            complexity = aux.get('complexity_map', [])
            if index < len(complexity) and complexity[index] is not None:
                values['complexity'] = complexity[index]
            for name, value in values.items():
                value = value.detach().double()
                stat = record['measurements'].setdefault(name, dict(n=0, sum=0., square_sum=0., zero=0, fallback_value=0))
                stat['n'] += value.numel()
                stat['sum'] += value.sum().item()
                stat['square_sum'] += value.square().sum().item()
                stat['zero'] += (value == 0).sum().item()
                stat['fallback_value'] += (value == .5).sum().item()

    def result(self, target=None):
        levels = {}
        for key, r in self.levels.items():
            levels[key] = dict(min_bits=r['min'], mean_bits=r['sum'] / r['tiles'],
                               max_bits=r['max'], saturated_max_fraction=r['saturated'] / r['tiles'],
                               element_mean_bits=r['element_sum'] / r['elements'],
                               histogram=r['histogram'], tiles=r['tiles'], descriptors={},
                               qdq_tensor_bytes_per_image=r['qdq_bytes'] / r['images'],
                               bit_map_tensor_bytes_per_image=r['map_bytes'] / r['images'],
                               ideal_bit_payload_bytes_per_image=r['element_sum'] / (8 * r['images']))
            for name, s in r['measurements'].items():
                mean = s['sum'] / s['n']
                stat = dict(mean=mean, std=math.sqrt(max(0., s['square_sum'] / s['n'] - mean ** 2)),
                            zero_fraction=s['zero'] / s['n'])
                if name == 'fractal':
                    # This is the frequency of D_f/2 == 0.5, not a proof that
                    # every such tile took the fallback branch.
                    stat['fallback_value_fraction'] = s['fallback_value'] / s['n']
                if name == 'complexity':
                    levels[key]['complexity'] = stat
                else:
                    levels[key]['descriptors'][name] = stat
        if not levels:
            return {}
        policy = sum(r['mean_bits'] for r in levels.values()) / len(levels)
        element_mean = sum(r['element_sum'] for r in self.levels.values()) / sum(r['elements'] for r in self.levels.values())
        return dict(policy_mean_bits=policy, element_mean_bits=element_mean,
                    target_bits=target, target_gap=None if target is None else policy - target,
                    levels=levels)
