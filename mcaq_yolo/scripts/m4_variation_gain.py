"""
M4-3: Within-image complexity variation vs. MCAQ gain (analysis figure).

Directly tests the paper's central cross-dataset hypothesis: the gain of
spatial allocation over uniform quantization grows with WITHIN-image
complexity variation. Bins images by the std of their tile complexities and
plots the per-bin mean AP gain (spatial-trained vs uniform-trained models,
each in its own training configuration) with bootstrap CIs.

Usage (full scale):
    python -m mcaq_yolo.scripts.m4_variation_gain \
        --ckpt-spatial outputs/<spatial>/best.pt \
        --ckpt-uniform outputs/<uniform>/best.pt \
        --data <dataset.yaml> --val-rel images/val --device cuda

CPU prototype (coco128): see defaults.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcaq_yolo.train import Trainer
from mcaq_yolo.utils.evaluation import (
    compute_map, extract_targets_per_image, _decode_outputs,
)
from mcaq_yolo.core.bit_allocation import ComplexityToBitMappingNetwork as MapperBase


class ConstantMapper(nn.Module):
    def __init__(self, bits): super().__init__(); self.bits = float(bits)
    def forward(self, c, temperature=1.0, return_continuous=False):
        return torch.full_like(MapperBase._normalize_complexity_shape(c), self.bits)


def per_image_records(model, loader, device, quantize=True):
    """Per-image: AP@0.5 (single-image mAP) + tile-complexity std (P3 scale)."""
    recs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].to(device)
            if imgs.dtype == torch.uint8:
                imgs = imgs.float() / 255.0
            B, _, H, W = imgs.shape
            outputs, aux = model(imgs, temperature=1.0, return_aux=True,
                                 quantize=quantize)
            dets = _decode_outputs(outputs)
            tgts = extract_targets_per_image(batch, B, W, H)
            cmaps = aux.get("complexity_map") or []
            cstd = (cmaps[0].float().reshape(B, -1).std(dim=1)
                    if cmaps else torch.zeros(B))
            for i in range(B):
                m = compute_map([dets[i].cpu()], [tgts[i]],
                                iou_thresholds=[0.5])
                recs.append({"ap50": m["mAP@0.5"],
                             "cstd": float(cstd[i])})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-spatial", default="outputs/coco128_v4/best.pt")
    ap.add_argument("--ckpt-uniform", default="outputs/coco128_v3/best.pt")
    ap.add_argument("--uniform-bits", type=float, default=4.0)
    ap.add_argument("--data", default="coco128_local.yaml")
    ap.add_argument("--train-rel", default="images/train2017")
    ap.add_argument("--val-rel", default="images/train2017")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--out-json", default="/tmp/m4_variation_gain.json")
    ap.add_argument("--out-fig", default="/tmp/m4_variation_gain.png")
    args = ap.parse_args()

    def build(ckpt):
        cfg = {
            "epochs": 1, "batch_size": args.batch, "device": args.device,
            "output_dir": str(Path(ckpt).parent),
            "data": {"yaml_path": args.data, "train": args.train_rel,
                     "val": args.val_rel, "num_workers": 0, "imgsz": 640},
            "model": {"teacher_path": "yolov8n.pt", "num_classes": 80,
                      "name": "yolov8n"},
            "quantization": {"min_bits": 2, "max_bits": 8, "target_bits": 4.0,
                             "grid_size": args.grid, "bit_mapping": "linear",
                             "normalize_complexity": False},
            "curriculum": {"enabled": True}, "distillation": {"enabled": False},
            "training": {"amp": False},
        }
        t = Trainer(cfg)
        t.model.load_state_dict(torch.load(ckpt, map_location=args.device),
                                strict=False)
        t.model.eval()
        return t

    print("[1/2] spatial-trained model (native allocation)...", flush=True)
    ts = build(args.ckpt_spatial)
    recs_s = per_image_records(ts.model, ts.val_loader, args.device)

    print("[2/2] uniform-trained model (uniform bits)...", flush=True)
    tu = build(args.ckpt_uniform)
    tu.model.bit_mapper = ConstantMapper(args.uniform_bits)
    recs_u = per_image_records(tu.model, tu.val_loader, args.device)

    # 같은 로더 순서 → 이미지 단위 짝 맞춤
    cstd = np.array([r["cstd"] for r in recs_s])
    gain = np.array([a["ap50"] - b["ap50"] for a, b in zip(recs_s, recs_u)])

    # 분위수 빈 + per-bin 평균 이득 (cluster=image라 단순 bootstrap)
    qs = np.quantile(cstd, np.linspace(0, 1, args.bins + 1))
    qs[-1] += 1e-9
    rows, rng = [], np.random.default_rng(0)
    for k in range(args.bins):
        sel = (cstd >= qs[k]) & (cstd < qs[k + 1])
        g = gain[sel]
        boots = [rng.choice(g, size=g.size, replace=True).mean()
                 for _ in range(2000)] if g.size else [0.0]
        rows.append({"bin": k, "cstd_lo": float(qs[k]), "cstd_hi": float(qs[k + 1]),
                     "n": int(g.size), "gain_mean": float(g.mean()) if g.size else 0.0,
                     "ci_lo": float(np.percentile(boots, 2.5)),
                     "ci_hi": float(np.percentile(boots, 97.5))})
        print(f"bin{k}: cstd [{qs[k]:.5g},{qs[k+1]:.5g}) n={g.size} "
              f"gain {rows[-1]['gain_mean']:+.4f} "
              f"[{rows[-1]['ci_lo']:+.4f},{rows[-1]['ci_hi']:+.4f}]", flush=True)

    from scipy.stats import spearmanr
    rho, p = spearmanr(cstd, gain)
    print(f"Spearman(cstd, gain) = {rho:.3f} (p={p:.3g}, n={len(gain)})")

    json.dump({"rows": rows, "spearman_rho": float(rho), "p": float(p)},
              open(args.out_json, "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(args.bins)
    means = [r["gain_mean"] for r in rows]
    err = [[r["gain_mean"] - r["ci_lo"] for r in rows],
           [r["ci_hi"] - r["gain_mean"] for r in rows]]
    plt.figure(figsize=(6, 4))
    plt.bar(x, means, yerr=err, capsize=4,
            color=plt.cm.viridis(np.linspace(0.2, 0.8, args.bins)))
    plt.axhline(0, color="k", lw=0.8)
    plt.xticks(x, [f"Q{k+1}" for k in range(args.bins)])
    plt.xlabel("Within-image complexity variation (quartiles of tile std)")
    plt.ylabel("AP@0.5 gain (spatial − uniform)")
    plt.title(f"Gain vs. complexity variation (ρ={rho:.2f}, p={p:.2g})")
    plt.tight_layout(); plt.savefig(args.out_fig, dpi=160)
    print(f"saved -> {args.out_json}, {args.out_fig}")


if __name__ == "__main__":
    main()
