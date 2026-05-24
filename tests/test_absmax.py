"""
Tests for Absmax Quantization
================================

Run with:  pytest tests/test_absmax.py -v

What we're testing:
- Round-trip accuracy: dequantize(quantize(x)) ≈ x
- Value range: quantized values fit in [-127, 127]
- Edge cases: zeros, single values, negative-only
- Scale is correct
- Per-channel version gives different scales per row
"""

import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quantization.absmax import absmax_quantize, absmax_dequantize, absmax_quantize_per_channel


# ---- Round-trip tests ----

def test_roundtrip_basic():
    """After quantize → dequantize, values should be close to original."""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    q, scale = absmax_quantize(x)
    x_recovered = absmax_dequantize(q, scale)
    # Should be very close (within one quantization step)
    assert torch.allclose(x, x_recovered, atol=scale + 1e-6), \
        f"Round-trip error too large. Original: {x}, Recovered: {x_recovered}"


def test_roundtrip_random_tensor():
    """Works on random weight-like tensor."""
    torch.manual_seed(42)
    x = torch.randn(64, 64) * 0.1  # typical weight magnitude
    q, scale = absmax_quantize(x)
    x_recovered = absmax_dequantize(q, scale)
    # Max absolute error should be at most one scale step
    max_error = (x - x_recovered).abs().max().item()
    assert max_error <= scale * 1.01, f"Max error {max_error:.6f} exceeds scale {scale:.6f}"


def test_roundtrip_int4():
    """Works with 4-bit quantization (range -7..7)."""
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    q, scale = absmax_quantize(x, bits=4)
    x_recovered = absmax_dequantize(q, scale)
    # INT4 is coarser — allow larger error
    max_error = (x - x_recovered).abs().max().item()
    assert max_error <= scale * 1.5, f"INT4 round-trip error too large: {max_error}"


# ---- Output range tests ----

def test_quantized_range_int8():
    """Quantized values must be in [-127, 127]."""
    x = torch.randn(1000)
    q, _ = absmax_quantize(x, bits=8)
    assert q.min().item() >= -127
    assert q.max().item() <= 127


def test_quantized_range_int4():
    """Quantized values for INT4 must be in [-7, 7]."""
    x = torch.randn(1000)
    q, _ = absmax_quantize(x, bits=4)
    assert q.min().item() >= -7
    assert q.max().item() <= 7


def test_output_dtype():
    """Quantized tensor should be int8."""
    x = torch.randn(10)
    q, _ = absmax_quantize(x)
    assert q.dtype == torch.int8, f"Expected int8, got {q.dtype}"


def test_scale_positive():
    """Scale should always be positive."""
    x = torch.randn(100)
    _, scale = absmax_quantize(x)
    assert scale > 0, f"Scale should be positive, got {scale}"


# ---- Edge cases ----

def test_all_zeros():
    """All-zero tensor should not crash and should recover zeros."""
    x = torch.zeros(10)
    q, scale = absmax_quantize(x)
    x_recovered = absmax_dequantize(q, scale)
    assert torch.all(x_recovered == 0), "Zero tensor should recover as zeros"


def test_single_value():
    """Single non-zero value."""
    x = torch.tensor([3.14])
    q, scale = absmax_quantize(x)
    x_recovered = absmax_dequantize(q, scale)
    assert x_recovered.shape == x.shape


def test_negative_only():
    """Tensor with only negative values."""
    x = torch.tensor([-5.0, -3.0, -1.0])
    q, scale = absmax_quantize(x)
    # Negative values should quantize to negative integers
    assert q.min().item() < 0, "Negative values should produce negative quantized values"
    x_recovered = absmax_dequantize(q, scale)
    assert torch.allclose(x, x_recovered, atol=scale + 1e-6)


def test_symmetric_output():
    """For perfectly symmetric input, max value should map to ±127."""
    x = torch.tensor([-2.0, 0.0, 2.0])
    q, scale = absmax_quantize(x)
    # -2.0 should map to -127, +2.0 should map to +127
    assert q[0].item() == -127
    assert q[2].item() == 127


# ---- Per-channel tests ----

def test_per_channel_different_scales():
    """Each row should get a different scale."""
    # Row 0 has values in [-1, 1], row 1 has values in [-100, 100]
    x = torch.tensor([
        [0.1, -0.5, 0.3],
        [50.0, -80.0, 100.0],
    ], dtype=torch.float32)
    q, scales = absmax_quantize_per_channel(x)
    assert scales.shape == (2, 1), f"Expected scales shape (2, 1), got {scales.shape}"
    # Row 1 should have a much larger scale
    assert scales[1, 0] > scales[0, 0], "Row 1 (larger values) should have larger scale"


def test_per_channel_shape():
    """Per-channel quantization preserves shape."""
    x = torch.randn(8, 64)
    q, scales = absmax_quantize_per_channel(x)
    assert q.shape == x.shape
    assert scales.shape == (8, 1)


def test_quantization_error_measure():
    """
    Demonstrate quantization error = dequant(quant(x)) - x.
    Error should be small relative to the original values.
    """
    torch.manual_seed(0)
    x = torch.randn(256, 256) * 0.2
    q, scale = absmax_quantize(x)
    x_recovered = absmax_dequantize(q, scale)

    error = (x - x_recovered).abs()
    relative_error = error.mean().item() / x.abs().mean().item()

    print(f"\nQuantization error stats:")
    print(f"  Mean absolute error: {error.mean().item():.6f}")
    print(f"  Max absolute error:  {error.max().item():.6f}")
    print(f"  Relative error:      {relative_error:.4f} ({relative_error*100:.2f}%)")

    # Relative error should be small (typically < 1% for INT8)
    assert relative_error < 0.05, f"Relative error {relative_error:.4f} too large"
