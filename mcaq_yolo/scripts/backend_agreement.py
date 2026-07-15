"""
Quantify GPU-surrogate vs cv2-reference agreement on a REAL dataset
(review recommendation: the paper describes Eq.21-24 but training uses
tensorized surrogates — report the correlation instead of leaving the gap
implicit).

For N images it computes, per metric (fractal/texture/gradient/edge/contour)
and for the fused MLP complexity map, tile-level Pearson r, Spearman rho and
the per-backend means, then prints a table and optionally writes JSON.

Usage:
    python -m mcaq_yolo.scripts.backend_agreement \
        --data path/to/data.yaml --split val --n 64 --imgsz 640 \
        [--json out.json] [--legacy]

`--legacy` re-enables the pre-review surrogate implementations
(canny_impl='legacy', binarize_impl='otsu', contour_components=False) so the
improvement of the parity patch can itself be measured.
"""

import argparse
import json

import numpy as np
import torch

from ..core.morphology import MorphologicalComplexityAnalyzer

METRICS = ["fractal", "texture", "gradient", "edge", "contour"]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def run(data_yaml: str, split: str, n: int, imgsz: int,
        device: str, legacy: bool):
    # Local imports keep the module importable without ultralytics.
    from types import SimpleNamespace
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import build_yolo_dataset

    data = check_det_dataset(data_yaml)
    cfg = get_cfg(overrides={"imgsz": imgsz, "task": "detect", "mode": "val"})
    ds = build_yolo_dataset(cfg=cfg, img_path=data[split], batch=1,
                            data=data, mode="val", rect=False, stride=32)

    kw = dict(device=device)
    if legacy:
        kw.update(canny_impl="legacy", binarize_impl="otsu",
                  contour_components=False)
    ana = MorphologicalComplexityAnalyzer(**kw).to(device).eval()

    per_metric = {m: {"gpu": [], "cv2": []} for m in METRICS}
    fused = {"gpu": [], "cv2": []}

    n = min(n, len(ds))
    for i in range(n):
        img = ds[i]["img"].float().unsqueeze(0).to(device)
        if img.max() > 1.5:
            img = img / 255.0
        for backend in ("gpu", "cv2"):
            ana.metric_backend = backend
            phi, _ = ana.compute_phi_tiles(img)
            for j, m in enumerate(METRICS):
                per_metric[m][backend].append(
                    phi[0, ..., j].reshape(-1).cpu().numpy())
            c = ana(img)  # fused map through the SAME MLP
            fused[backend].append(c.reshape(-1).cpu().numpy())

    rows = []
    for m in METRICS:
        a = np.concatenate(per_metric[m]["gpu"])
        b = np.concatenate(per_metric[m]["cv2"])
        rows.append((m, _pearson(a, b), _spearman(a, b),
                     float(a.mean()), float(b.mean())))
    fa = np.concatenate(fused["gpu"])
    fb = np.concatenate(fused["cv2"])
    rows.append(("fused_C", _pearson(fa, fb), _spearman(fa, fb),
                 float(fa.mean()), float(fb.mean())))

    print(f"\nBackend agreement over {n} images "
          f"({'LEGACY' if legacy else 'cv2compat'} surrogates), "
          f"{fa.size // n} tiles/image:")
    print(f"{'metric':<10}{'pearson':>9}{'spearman':>10}"
          f"{'mean_gpu':>10}{'mean_cv2':>10}")
    for m, r, s, mg, mc in rows:
        print(f"{m:<10}{r:>9.3f}{s:>10.3f}{mg:>10.3f}{mc:>10.3f}")
    return {m: {"pearson": r, "spearman": s, "mean_gpu": mg, "mean_cv2": mc}
            for m, r, s, mg, mc in rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="dataset yaml")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", default=None, help="optional output json path")
    ap.add_argument("--legacy", action="store_true",
                    help="use pre-review surrogate implementations")
    args = ap.parse_args(argv)

    out = run(args.data, args.split, args.n, args.imgsz, args.device, args.legacy)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
