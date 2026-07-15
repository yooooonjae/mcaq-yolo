"""
Smoke tests for MCAQ-YOLO — run these FIRST on a machine with torch installed
(they were authored in an environment without torch, so nothing here has been
executed yet):

    cd /path/to/yolo            # parent of mcaq_yolo/
    python -m pytest mcaq_yolo/tests/test_smoke.py -v

CPU-friendly: no GPU, no ultralytics weights needed except for the final
(optional) end-to-end test, which is skipped unless ultralytics is available.
"""

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Allow running from the repo parent without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcaq_yolo.core.morphology import MorphologicalComplexityAnalyzer
from mcaq_yolo.core.bit_allocation import ComplexityToBitMappingNetwork
from mcaq_yolo.core.quantization import LearnedSoftMask, SpatialAdaptiveQuantization
from mcaq_yolo.core.curriculum import CurriculumScheduler


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H", [640, 80, 40, 20])
def test_phi_tiles_shapes(H):
    """compute_phi_tiles must return (B, ht, wt, 8) for all backbone sizes
    (incl. the formerly-crashing non-power-of-two tile cases)."""
    a = MorphologicalComplexityAnalyzer(device="cpu")
    x = torch.rand(2, 3, H, H)
    phi, detailed = a.compute_phi_tiles(x)
    tile = a._tile_size(H)
    ht = H // tile
    assert phi.shape == (2, ht, ht, 8), phi.shape
    assert set(detailed) == {"fractal", "texture", "gradient", "edge", "contour"}
    # Power-of-two tiles (Algorithm 1) and >= 2 dyadic scales (Algorithm 2)
    assert tile >= 4 and (tile & (tile - 1)) == 0
    # All metrics normalized
    assert float(phi.min()) >= 0.0 and float(phi.max()) <= 1.0 + 1e-5


def test_analyzer_forward_range_and_grad():
    a = MorphologicalComplexityAnalyzer(device="cpu")
    a.train()
    x = torch.rand(2, 16, 80, 80)
    c = a(x)
    assert c.dim() == 3 and 0.0 <= float(c.min()) and float(c.max()) <= 1.0
    # Gradients must reach the complexity MLP (phi itself is no-grad side info)
    c.sum().backward()
    grads = [p.grad for p in a.complexity_mlp.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


def test_score_image_deterministic():
    a = MorphologicalComplexityAnalyzer(device="cpu")
    x = torch.rand(1, 3, 160, 160)
    s1, s2 = a.score_image(x), a.score_image(x)
    assert torch.allclose(s1, s2)
    assert 0.0 <= float(s1) <= 1.0


# ---------------------------------------------------------------------------
# Bit allocation (Eq.13-17, Algorithm 3 line 13)
# ---------------------------------------------------------------------------

def test_bit_mapper_range_and_temperature():
    m = ComplexityToBitMappingNetwork(min_bits=2, max_bits=8)
    m.eval()
    c = torch.rand(2, 8, 8)
    b = m(c, temperature=1.0)
    assert b.shape == (2, 8, 8)
    assert float(b.min()) >= 2.0 and float(b.max()) <= 8.0
    assert torch.allclose(b, torch.round(b))  # integer bits at eval
    # alpha_t = 10 must saturate everything at bmax (warm-up semantics)
    b10 = m(c, temperature=10.0)
    assert torch.allclose(b10, torch.full_like(b10, 8.0))


def test_bit_mapper_gradient_through_clamp_and_round():
    """clamp-STE + round-STE: gradient must reach the MLP even when saturated."""
    m = ComplexityToBitMappingNetwork(min_bits=2, max_bits=8)
    m.train()
    c = torch.rand(2, 8, 8, requires_grad=False)
    b = m(c, temperature=10.0)  # fully saturated at 8
    (b.mean() - 4.0).pow(2).backward()  # Lbit-style loss
    grads = [p.grad for p in m.mapping_network.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads), \
        "clamp killed the gradient — STE regression"


# ---------------------------------------------------------------------------
# Quantization (Eq.19)
# ---------------------------------------------------------------------------

def test_fractional_bit_gradient_to_bit_map():
    """Ldet-style losses must be differentiable w.r.t. the continuous bit map."""
    q = SpatialAdaptiveQuantization(smooth_transitions=False)
    q.train()
    x = torch.randn(1, 4, 16, 16)
    bit_map = torch.full((1, 4, 4), 4.5, requires_grad=True)
    y = q(x, bit_map, training=True)
    assert y.shape == x.shape
    y.pow(2).mean().backward()
    assert bit_map.grad is not None and float(bit_map.grad.abs().sum()) > 0


def test_learned_soft_mask_near_identity_init():
    mask = LearnedSoftMask()
    x = torch.randn(2, 8, 32, 32)
    bit_map = torch.full((2, 4, 4), 4.0)
    m = mask(bit_map, x)
    assert m.shape == (2, 1, 32, 32)
    # Near-identity at init (~0.982), including at the borders (replicate pad)
    assert float(m.min()) > 0.9, float(m.min())
    # And trainable: gradient reaches BOTH convs (not gradient-dead)
    m.sum().backward()
    g_first = mask.net[0].weight.grad
    assert g_first is not None and float(g_first.abs().sum()) > 0


def test_calibration_freeze():
    q = SpatialAdaptiveQuantization(smooth_transitions=False)
    q.train()
    x = torch.randn(2, 4, 16, 16)
    bit_map = torch.full((2, 4, 4), 4.0)
    _ = q(x, bit_map, training=True)
    assert q.running_min is not None
    frozen_min = q.running_min.clone()
    q.freeze_calibration()
    _ = q(torch.randn(2, 4, 16, 16) * 100, bit_map, training=True)
    assert torch.allclose(q.running_min, frozen_min), "stats moved after freeze"


# ---------------------------------------------------------------------------
# Curriculum (Fig.3 / Algorithm 3 / Table X)
# ---------------------------------------------------------------------------

def test_curriculum_schedule():
    cs = CurriculumScheduler(warmup_epochs=20, transition_epochs=50, total_epochs=300)
    assert cs.get_stage(1) == 1 and cs.get_stage(20) == 1
    assert cs.get_stage(21) == 2 and cs.get_stage(50) == 2
    assert cs.get_stage(51) == 3
    # alpha_t = 1 + 9*exp(-5t/T)
    assert math.isclose(cs.get_temperature(0), 10.0, rel_tol=1e-6)
    assert cs.get_temperature(300) < 1.1
    # tau ramps 0.2 -> 1.0 over warm-up (Algorithm 3 line 5)
    assert math.isclose(cs.get_complexity_threshold(0), 0.2, rel_tol=1e-6)
    assert math.isclose(cs.get_complexity_threshold(20), 1.0, rel_tol=1e-6)
    w = cs.get_loss_weights(0)
    assert math.isclose(w['bit_budget'], 0.01, rel_tol=1e-6)
    # Smoothness ramps: 0 during warm-up (quantization bypassed), full lambda2
    # from the end of the transition stage (Codex review #4)
    assert w['smoothness'] == 0.0
    assert math.isclose(cs.get_loss_weights(50)['smoothness'], 0.1, rel_tol=1e-6)
    assert math.isclose(w['distillation'], 0.5, rel_tol=1e-6)
    # Bit-target schedule (Codex review #1): 8 during warm-up, ~4 by the end
    assert math.isclose(cs.get_target_bits(0), 8.0, rel_tol=1e-6)
    assert cs.get_target_bits(300) < 4.5


# ---------------------------------------------------------------------------
# End-to-end (needs ultralytics + downloads yolov8n.pt on first run)
# ---------------------------------------------------------------------------

def test_model_forward_end_to_end():
    pytest.importorskip("ultralytics")
    from mcaq_yolo.models.mcaq_yolo import MCAQYOLO

    model = MCAQYOLO(model_name="yolov8n", pretrained=True, device="cpu")
    model.train()
    x = torch.rand(1, 3, 640, 640)
    outputs, aux = model(x, temperature=1.0, quantize=True)
    assert len(aux['bit_map']) == 3, "expected one bit map per backbone scale"
    assert 2.0 <= float(aux['avg_bits']) <= 8.0
    # Backbone discovery: no duplicate hooks (off-by-one regression guard)
    idxs = model.backbone_out_indices
    assert len(idxs) == len(set(idxs)) == 3, idxs


def test_linear_bit_mapper_spatial_variance():
    """Paper's 'Linear mapping' ablation: bit maps must reflect the relative
    spatial structure of the complexity map (2..8 spread, not a constant)."""
    from mcaq_yolo.core.bit_allocation import LinearBitMapper
    m = LinearBitMapper(min_bits=2, max_bits=8)
    c = torch.linspace(0, 1, 16).reshape(1, 4, 4) * 0.05 + 0.4  # narrow absolute range
    b = m(c, temperature=1.0)
    assert float(b.min()) == 2.0 and float(b.max()) == 8.0  # normalization spreads it
    assert torch.unique(b).numel() >= 5
