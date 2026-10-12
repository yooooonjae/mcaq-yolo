"""Metric contracts for the reference and converted bbox evaluation protocols."""

import sys

import pytest
import torch


@pytest.fixture
def evaluate():
    from mcaq_yolo.utils.protocol_evaluation import evaluate_detections

    return evaluate_detections


@pytest.fixture
def coco_evaluate(evaluate):
    pytest.importorskip("pycocotools.cocoeval")
    return evaluate


def test_reference_reports_iou_average_without_claiming_size_ap(evaluate):
    # IoU=.62 passes .50, .55, .60, but none of the other seven thresholds.
    result = evaluate(
        [torch.tensor([[0, 0, 6.2, 10, .9, 0]])],
        [torch.tensor([[0, 0, 0, 10, 10]])], [(100, 100)],
    )
    assert result["mAP50"] == pytest.approx(1)
    assert result["mAP50_95"] == pytest.approx(.3)
    assert all(result[key] is None for key in ("AP_small", "AP_medium", "AP_large"))
    assert result["protocol"] == "reference_voc"


def test_reference_preserves_detected_only_class_penalty(evaluate):
    result = evaluate(
        [torch.tensor([[0, 0, 20, 20, .9, 0], [0, 0, 20, 20, .8, 9]])],
        [torch.tensor([[0, 0, 0, 20, 20]])], [(100, 100)],
    )
    assert result["mAP50"] == pytest.approx(.5)


def test_reference_empty_predictions_are_zero(evaluate):
    result = evaluate([torch.empty(0, 6)], [torch.tensor([[0, 0, 0, 20, 20]])], [(100, 100)])
    assert result["mAP50"] == result["mAP50_95"] == 0


def test_coco_perfect_detection_and_undefined_bins(coco_evaluate):
    result = coco_evaluate(
        [torch.tensor([[0, 0, 20, 20, .9, 7]])],
        [torch.tensor([[7, 0, 0, 20, 20]])], [(100, 100)], protocol="coco_bbox",
    )
    assert result["mAP50"] == pytest.approx(1)
    assert result["mAP50_95"] == pytest.approx(1)
    assert result["AP_small"] == pytest.approx(1)
    assert result["AP_medium"] is result["AP_large"] is None
    assert result["protocol"] == "coco_bbox"
    assert result["protocol_details"]["annotation_source"] == "converted_yolo_boxes"
    assert result["protocol_details"]["official_coco_annotations"] is False


def test_coco_averages_all_ten_iou_thresholds(coco_evaluate):
    result = coco_evaluate(
        [torch.tensor([[0, 0, 6.2, 10, .9, 0]])],
        [torch.tensor([[0, 0, 0, 10, 10]])], [(100, 100)], protocol="coco_bbox",
    )
    assert result["mAP50"] == pytest.approx(1)
    assert result["mAP50_95"] == pytest.approx(.3)


def test_coco_uses_101_recall_points(coco_evaluate):
    result = coco_evaluate(
        [torch.tensor([[0, 0, 20, 20, .9, 0]])],
        [torch.tensor([[0, 0, 0, 20, 20], [0, 40, 40, 60, 60]])],
        [(100, 100)], protocol="coco_bbox",
    )
    # Recall is exactly .5, so 51 of COCO's 101 recall samples have precision 1.
    assert result["mAP50_95"] == pytest.approx(51 / 101)


def test_coco_reports_each_area_bin(coco_evaluate):
    result = coco_evaluate(
        [torch.tensor([[0, 0, size, size, .9, 0]]) for size in (20, 64, 120)],
        [torch.tensor([[0, 0, 0, size, size]]) for size in (20, 64, 120)],
        [(200, 200)] * 3, protocol="coco_bbox",
    )
    for key in ("AP_small", "AP_medium", "AP_large"):
        assert result[key] == pytest.approx(1)


def test_coco_does_not_silently_clip_box_area(coco_evaluate):
    result = coco_evaluate(
        [torch.tensor([[-20, -20, 20, 20, .9, 0]])],
        [torch.tensor([[0, -20, -20, 20, 20]])], [(40, 40)], protocol="coco_bbox",
    )
    assert result["AP_medium"] == pytest.approx(1)
    assert result["AP_small"] is None


def test_coco_empty_predictions_are_zero_for_present_gt(coco_evaluate):
    result = coco_evaluate(
        [torch.empty(0, 6)], [torch.tensor([[0, 0, 0, 20, 20]])],
        [(100, 100)], protocol="coco_bbox",
    )
    assert result["mAP50"] == result["mAP50_95"] == result["AP_small"] == 0
    assert result["AP_medium"] is result["AP_large"] is None


@pytest.mark.parametrize("has_detections", [False, True])
def test_coco_no_gt_is_undefined(coco_evaluate, has_detections):
    det = torch.tensor([[0, 0, 20, 20, .9, 0]]) if has_detections else torch.empty(0, 6)
    result = coco_evaluate([det], [torch.empty(0, 5)], [(100, 100)], protocol="coco_bbox")
    for key in ("mAP50", "mAP50_95", "AP_small", "AP_medium", "AP_large"):
        assert result[key] is None


def test_coco_ignores_categories_without_any_gt(coco_evaluate):
    result = coco_evaluate(
        [torch.tensor([[0, 0, 20, 20, .9, 0], [0, 0, 20, 20, .99, 9]])],
        [torch.tensor([[0, 0, 0, 20, 20]])], [(100, 100)], protocol="coco_bbox",
    )
    assert result["mAP50_95"] == pytest.approx(1)


def test_coco_counts_false_positives_on_empty_images(coco_evaluate):
    result = coco_evaluate(
        [torch.tensor([[0, 0, 20, 20, .99, 0]]), torch.tensor([[0, 0, 20, 20, .9, 0]])],
        [torch.empty(0, 5), torch.tensor([[0, 0, 0, 20, 20]])],
        [(100, 100), (100, 100)], protocol="coco_bbox",
    )
    assert result["mAP50_95"] == pytest.approx(.5)


def test_coco_caps_evaluation_at_100_detections(coco_evaluate):
    det = torch.tensor([[40, 40, 60, 60, .99, 0]] * 100 + [[0, 0, 20, 20, .9, 0]])
    result = coco_evaluate(
        [det], [torch.tensor([[0, 0, 0, 20, 20]])], [(100, 100)], protocol="coco_bbox",
    )
    assert result["mAP50_95"] == 0


def test_missing_optional_coco_dependency_has_actionable_error(evaluate, monkeypatch):
    monkeypatch.setitem(sys.modules, "pycocotools.coco", None)
    with pytest.raises(ImportError, match="pycocotools"):
        evaluate([], [], [], protocol="coco_bbox")


def test_unknown_protocol_is_rejected(evaluate):
    with pytest.raises(ValueError, match="protocol"):
        evaluate([], [], [], protocol="coco")


@pytest.mark.parametrize("sizes", [[], [(0, 100)], [(100, float("nan"))], [(100.5, 100)], [(100,)]])
def test_image_sizes_must_match_and_be_positive_integer_pairs(evaluate, sizes):
    with pytest.raises(ValueError, match="image_sizes"):
        evaluate([torch.empty(0, 6)], [torch.empty(0, 5)], sizes)


def test_targets_must_match_detection_image_count(evaluate):
    with pytest.raises(ValueError, match="same number"):
        evaluate([torch.empty(0, 6)], [], [(100, 100)])


@pytest.mark.parametrize("row", [
    [0, 0, 0, 20, .9, 0], [20, 0, 0, 20, .9, 0],
    [0, 0, float("inf"), 20, .9, 0], [0, 0, 20, 20, float("nan"), 0],
    [0, 0, 20, 20, -.1, 0], [0, 0, 20, 20, 1.1, 0],
    [0, 0, 20, 20, .9, -.1], [0, 0, 20, 20, .9, 1.5],
])
def test_invalid_detections_are_rejected(evaluate, row):
    with pytest.raises(ValueError, match="detections"):
        evaluate([torch.tensor([row])], [torch.empty(0, 5)], [(100, 100)])


@pytest.mark.parametrize("target", [
    torch.empty(0, 4), torch.tensor([[0, 0, 0, 0, 20]]),
    torch.tensor([[1.5, 0, 0, 20, 20]]), torch.tensor([[0, 0, 0, 20, float("nan")]]),
])
def test_invalid_targets_are_rejected(evaluate, target):
    with pytest.raises(ValueError, match="targets"):
        evaluate([torch.empty(0, 6)], [target], [(100, 100)])
