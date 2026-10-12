"""Explicit AP protocols for decoded pixel-space detections.

``reference_voc`` retains the repository's reference AP implementation.
``coco_bbox`` runs official COCOeval on converted YOLO box annotations. The
conversion cannot recover crowd/ignore flags, segmentation areas, or omitted
annotations, so these are not published COCO benchmark scores.
"""

from contextlib import redirect_stdout
from importlib.metadata import version
from io import StringIO
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def _validated_inputs(detections, targets, image_sizes):
    if len(detections) != len(targets) or len(detections) != len(image_sizes):
        raise ValueError("detections, targets, and image_sizes must have the same number of images")

    sizes = []
    for index, size in enumerate(image_sizes):
        try:
            if len(size) != 2:
                raise ValueError
            height, width = (float(value) for value in size)
            if any(not math.isfinite(value) or value <= 0 or not value.is_integer()
                   for value in (height, width)):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"image_sizes[{index}] must be a positive integer (height, width) pair") from None
        sizes.append((int(height), int(width)))

    normalized = []
    for name, entries, columns in (("detections", detections, 6), ("targets", targets, 5)):
        tensors = []
        for index, entry in enumerate(entries):
            label = f"{name}[{index}]"
            if not isinstance(entry, torch.Tensor) or entry.ndim != 2 or entry.shape[1] != columns:
                raise ValueError(f"{label} must be a tensor of shape (N, {columns})")
            if entry.is_complex():
                raise ValueError(f"{label} must contain real finite values")
            tensor = entry.detach().to(device="cpu", dtype=torch.float64)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{label} must contain finite values")
            boxes = tensor[:, :4] if name == "detections" else tensor[:, 1:5]
            dimensions = boxes[:, 2:4] - boxes[:, :2]
            if (dimensions <= 0).any() or not torch.isfinite(dimensions.prod(dim=1)).all():
                raise ValueError(f"{label} boxes must have positive finite width, height, and area")
            classes = tensor[:, 5] if name == "detections" else tensor[:, 0]
            if (classes < 0).any() or (classes != classes.round()).any():
                raise ValueError(f"{label} class IDs must be nonnegative integers")
            if name == "detections" and ((tensor[:, 4] < 0) | (tensor[:, 4] > 1)).any():
                raise ValueError(f"{label} confidence scores must be in [0, 1]")
            tensors.append(tensor)
        normalized.append(tensors)
    return normalized[0], normalized[1], sizes


def _coco_bbox(detections, targets, image_sizes):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError(
            "The coco_bbox evaluation protocol requires pycocotools. "
            "Install it in the active environment with: python -m pip install pycocotools"
        ) from exc

    images = [
        {"id": index, "height": height, "width": width}
        for index, (height, width) in enumerate(image_sizes, start=1)
    ]
    class_ids = sorted(
        {int(row[0]) for tensor in targets for row in tensor}
        | {int(row[5]) for tensor in detections for row in tensor}
    )
    categories = [{"id": class_id, "name": str(class_id)} for class_id in class_ids]
    annotations, results = [], []
    for image_id, (det, target) in enumerate(zip(detections, targets), start=1):
        for class_id, x1, y1, x2, y2 in target.tolist():
            width, height = x2 - x1, y2 - y1
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id,
                "category_id": int(class_id), "bbox": [x1, y1, width, height],
                "area": width * height, "iscrowd": 0,
            })
        for x1, y1, x2, y2, score, class_id in det.tolist():
            results.append({
                "image_id": image_id, "category_id": int(class_id),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score,
            })

    # COCO.loadRes indexes results[0], so empty predictions need an indexed,
    # genuinely empty result dataset, not a fabricated zero-confidence box.
    with redirect_stdout(StringIO()):
        ground_truth = COCO()
        ground_truth.dataset = {
            "images": images, "categories": categories, "annotations": annotations,
            "info": {"description": "Converted YOLO bbox-only annotations"},
        }
        ground_truth.createIndex()
        if results:
            predictions = ground_truth.loadRes(results)
        else:
            predictions = COCO()
            predictions.dataset = {"images": images, "categories": categories, "annotations": []}
            predictions.createIndex()
        evaluator = COCOeval(ground_truth, predictions, "bbox")
        # Keep COCOeval's official bbox defaults, including 101 recall points,
        # area ranges, category-aware matching, and maxDets=[1, 10, 100].
        if annotations:
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            stats = evaluator.stats
        else:
            stats = [-1.0] * 12

    def defined(index):
        return float(stats[index]) if stats[index] >= 0 else None

    return {
        "mAP50": defined(1), "mAP50_95": defined(0),
        "AP_small": defined(3), "AP_medium": defined(4), "AP_large": defined(5),
        "protocol": "coco_bbox",
        "protocol_details": {
            "evaluator": "pycocotools.COCOeval(bbox)",
            "evaluator_version": version("pycocotools"),
            "iou_thresholds": evaluator.params.iouThrs.tolist(),
            "recall_points": len(evaluator.params.recThrs),
            "max_detections": list(evaluator.params.maxDets),
            "area_ranges": dict(zip(evaluator.params.areaRngLbl, evaluator.params.areaRng)),
            "area_coordinates": "provided pixel coordinates",
            "area_source": "unclipped_bbox_width_times_height",
            "annotation_source": "converted_yolo_boxes",
            "official_coco_annotations": False,
            "crowd_and_ignore_flags_available": False,
            "categories_without_ground_truth": "excluded",
            "undefined_ap": None,
            "limitations": (
                "BBox-only converted-annotation protocol; all targets are non-crowd. "
                "YOLO boxes cannot recover crowd/ignore flags, segmentation areas, or omitted annotations. "
                "Size AP uses supplied pixel coordinates; provide original-image boxes for original-image bins. "
                "These results are not official published COCO benchmark scores."
            ),
        },
    }


def evaluate_detections(
    detections: List[torch.Tensor],
    targets: List[torch.Tensor],
    image_sizes: Sequence[Tuple[int, int]],
    protocol: str = "reference_voc",
) -> Dict:
    """Evaluate decoded detections with an explicitly named AP protocol.

    Args:
        detections: Per-image (N, 6): [x1, y1, x2, y2, confidence, class].
        targets: Per-image (M, 5): [class, x1, y1, x2, y2].
        image_sizes: Matching (height, width) in the boxes' pixel coordinate
            frame. For original-image size AP, restore BOTH detections and
            targets to original-image pixels before calling this function.
        protocol: ``reference_voc`` or ``coco_bbox``. The latter requires the
            optional pycocotools package; it never falls back silently.

    Boxes are validated but not clipped, rescaled, filtered, or suppressed.
    AP values are fractions, not percentages. Undefined COCO AP (no ground
    truth in a category/area) is None; existing GT with no detections is zero.
    Reference size AP is unavailable (None), and its legacy class averaging
    includes detected-only classes as zero, unlike COCOeval.
    """
    if protocol not in ("reference_voc", "coco_bbox"):
        raise ValueError("protocol must be 'reference_voc' or 'coco_bbox'")
    detections, targets, image_sizes = _validated_inputs(detections, targets, image_sizes)
    if protocol == "coco_bbox":
        return _coco_bbox(detections, targets, image_sizes)

    from .evaluation import compute_map

    thresholds = np.linspace(.5, .95, 10).tolist()
    metrics = compute_map(detections, targets, iou_thresholds=thresholds, interp="voc")
    return {
        "mAP50": metrics["mAP@0.5"], "mAP50_95": metrics["mAP@0.5:0.95"],
        "AP_small": None, "AP_medium": None, "AP_large": None,
        "protocol": "reference_voc",
        "protocol_details": {
            "evaluator": "mcaq_yolo.utils.evaluation.compute_map",
            "iou_thresholds": thresholds,
            "interpolation": "VOC all-point precision envelope",
            "categories_without_ground_truth": "included_as_zero",
            "area_coordinates": "provided pixel coordinates",
            "size_ap_available": False,
            "official_coco_annotations": False,
        },
    }
