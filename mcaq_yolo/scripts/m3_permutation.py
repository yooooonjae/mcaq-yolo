"""
M3-1: Post-hoc bit-placement permutation test (inference-only).

Question: given a trained model and a FIXED per-image bit histogram, does the
complexity-guided PLACEMENT of bits matter?
Expected ordering: MCAQ placement > random permutation > inverted placement.

Caveat (per Appendix C): with training-inference matching being decisive, this
measures the importance of placement GIVEN the learned representation.

Usage (full scale):
    python -m mcaq_yolo.scripts.m3_permutation \
        --checkpoint outputs/<run>/best.pt --data <dataset.yaml> \
        --train-rel images/train --val-rel images/val \
        --grid 16 --permutations 10 --device cuda

CPU prototype (coco128):
    python -m mcaq_yolo.scripts.m3_permutation \
        --checkpoint outputs/coco128_v4/best.pt \
        --data coco128_local.yaml --train-rel images/train2017 \
        --val-rel images/train2017 --permutations 10
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcaq_yolo.train import Trainer
from mcaq_yolo.utils.evaluation import evaluate_mcaq_yolo
from mcaq_yolo.core.bit_allocation import (
    ComplexityToBitMappingNetwork as MapperBase,
    LinearBitMapper,
)


class PermutedMapper(nn.Module):
    """Wraps a base mapper; permutes tile placement, preserving the per-image
    bit histogram exactly. mode='random' (seeded per image) or 'inverted'
    (highest complexity receives the lowest bits)."""

    def __init__(self, base: nn.Module, mode: str = "random", seed: int = 0):
        super().__init__()
        self.base = base
        self.mode = mode
        self.seed = seed

    def forward(self, c, temperature=1.0, return_continuous=False):
        b = self.base(c, temperature=temperature, return_continuous=return_continuous)
        cs = MapperBase._normalize_complexity_shape(c)
        B = b.shape[0]
        out = b.clone()
        for i in range(B):
            flat_b = b[i].reshape(-1)
            if self.mode == "random":
                # Deterministic per image: content-derived seed + run seed
                g = torch.Generator()
                g.manual_seed(self.seed * 1_000_003 + int(cs[i].sum().item() * 1e6) % (2**31))
                perm = torch.randperm(flat_b.numel(), generator=g)
                out[i] = flat_b[perm].reshape(b[i].shape)
            elif self.mode == "inverted":
                order_c = cs[i].reshape(-1).argsort()            # complexity ascending
                bits_desc = flat_b.sort(descending=True).values  # bits descending
                inv = torch.empty_like(flat_b)
                inv[order_c] = bits_desc                          # low C <- high bits
                out[i] = inv.reshape(b[i].shape)
            else:
                raise ValueError(self.mode)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-rel", default="images/train")
    ap.add_argument("--val-rel", default="images/val")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--permutations", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="/tmp/m3_permutation.json")
    args = ap.parse_args()

    cfg = {
        "epochs": 1, "batch_size": args.batch, "device": args.device,
        "output_dir": str(Path(args.checkpoint).parent),
        "data": {"yaml_path": args.data, "train": args.train_rel,
                 "val": args.val_rel, "num_workers": 0, "imgsz": 640},
        "model": {"teacher_path": "yolov8n.pt", "num_classes": 80, "name": "yolov8n"},
        "quantization": {"min_bits": 2, "max_bits": 8, "target_bits": 4.0,
                         "grid_size": args.grid, "bit_mapping": "linear",
                         "normalize_complexity": False},
        "curriculum": {"enabled": True}, "distillation": {"enabled": False},
        "training": {"amp": False},
    }
    t = Trainer(cfg)
    t.model.load_state_dict(torch.load(args.checkpoint, map_location=args.device),
                            strict=False)
    t.model.eval()
    base = t.model.bit_mapper

    results = {}

    def run(label, mapper):
        t.model.bit_mapper = mapper
        r = evaluate_mcaq_yolo(t.model, t.val_loader, device=args.device,
                               compute_complexity=False, quantize=True)
        results[label] = {"mAP50": r["mAP@0.5"], "mAP5095": r["mAP@0.5:0.95"],
                          "avg_bits": r["avg_bits"]}
        print(f">>> {label}: mAP@0.5 {r['mAP@0.5']:.4f} | avg {r['avg_bits']:.2f}",
              flush=True)

    run("mcaq_placement", base)
    perm_scores = []
    for k in range(args.permutations):
        run(f"random_perm_{k}", PermutedMapper(base, "random", seed=k))
        perm_scores.append(results[f"random_perm_{k}"]["mAP50"])
    run("inverted", PermutedMapper(base, "inverted"))

    import statistics
    summary = {
        "mcaq": results["mcaq_placement"]["mAP50"],
        "random_mean": statistics.mean(perm_scores),
        "random_std": statistics.stdev(perm_scores) if len(perm_scores) > 1 else 0.0,
        "inverted": results["inverted"]["mAP50"],
        "n_permutations": args.permutations,
    }
    results["_summary"] = summary
    json.dump(results, open(args.out, "w"), indent=2)
    print("\n===== M3-1 SUMMARY =====")
    print(f"MCAQ placement : {summary['mcaq']:.4f}")
    print(f"Random (n={args.permutations}) : {summary['random_mean']:.4f} ± {summary['random_std']:.4f}")
    print(f"Inverted       : {summary['inverted']:.4f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
