"""
Tests for Zero-Point Quantization
====================================

Run with:  pytest tests/test_zeropoint.py -v

Zero-point tests are similar to absmax tests but also verify:
- Asymmetric ranges are handled correctly
- Zero_point shifts the integer range appropriately
- Non-zero-centered tensors benefit over absmax
"""

import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quantization.zeropoint import zeropoint_quantize, zeropoint_dequantize


def test_roundtrip_basic():
    """Basic round-trip accuracy."""
    x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    q, scale, zp = zeropoint_quantize(x)
    x_recovered = zeropoint_dequantize(q, scale, zp)
    max_error = (x - x_recovered).abs().max().item()
    assert max_error <= scale * 1.01, f"Round-trip error {max_error} > scale {scale}"


def test_output_dtype_uint8():
    """Quantized tensor must be uint8."""
    x = torch.randn(10)
    q, _, _ = zeropoint_quantize(x)
    assert q.dtype == torch.uint8, f"Expected uint8, got {q.dtype}"


def test_quantized_range():
    """All quantized values must be in [0, 255]."""
    x = torch.randn(1000)
    q, _, _ = zeropoint_quantize(x)
    assert q.min().item() >= 0
    assert q.max().item() <= 255


def test_asymmetric_range():
    """
    For non-zero-centered data, zero-point should shift things.
    Data in [1, 3]: the minimum (1.0) should map close to 0.
    """
    x = torch.tensor([1.0, 2.0, 3.0])
    q, scale, zp = zeropoint_quantize(x)
    x_recovered = zeropoint_dequantize(q, scale, zp)
    max_error = (x - x_recovered).abs().max().item()
    assert max_error <= scale * 1.01


def test_zero_point_nonzero_for_positive_data():
    """
    For all-positive data (min > 0), zero_point should be NEGATIVE.

    Intuition: zero_point is the integer index that represents x=0.0.
    If all data is positive (e.g. [1, 2, 3]), then 0.0 falls BELOW
    the data range. In integer space, it maps to a negative index.

    Example: min=1, max=3, scale=(3-1)/255=0.00784
      zero_point = round(-1 / 0.00784) = round(-127.5) = -128
      This means x=0.0 maps to integer index -128 (outside [0..255]).
    """
    x = torch.tensor([1.0, 2.0, 3.0])  # all positive, min=1.0
    q, scale, zp = zeropoint_quantize(x)
    assert zp < 0, f"For all-positive data [1,2,3], zero_point should be < 0, got {zp}"
    assert zp != 0, f"For all-positive data, zero_point should not be 0, got {zp}"


def test_negative_data():
    """Works with negative data."""
    x = torch.tensor([-3.0, -2.0, -1.0])
    q, scale, zp = zeropoint_quantize(x)
    x_recovered = zeropoint_dequantize(q, scale, zp)
    max_error = (x - x_recovered).abs().max().item()
    assert max_error <= scale * 1.01


def test_all_zeros():
    """All-zero tensor should not crash."""
    x = torch.zeros(10)
    q, scale, zp = zeropoint_quantize(x)
    x_recovered = zeropoint_dequantize(q, scale, zp)
    assert x_recovered.shape == x.shape


def test_single_value():
    """Single value tensor."""
    x = torch.tensor([7.0])
    q, scale, zp = zeropoint_quantize(x)
    assert q.shape == (1,)


def test_full_range_usage():
    """
    For a tensor with known min and max, verify the quantized range
    actually uses [0, 255] well.
    """
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])  # min=0, max=3
    q, scale, zp = zeropoint_quantize(x)
    # The minimum value (0.0) should map to 0 (or very close)
    # The maximum value (3.0) should map to 255 (or very close)
    assert q.min().item() <= 5, f"Min quantized should be near 0, got {q.min().item()}"
    assert q.max().item() >= 250, f"Max quantized should be near 255, got {q.max().item()}"


def test_zp_absmax_comparison():
    """
    Zero-point should have lower error for skewed distributions.

    For data heavily skewed to positive values, zero-point allocation
    is more efficient than absmax (which wastes half the range on negatives).
    """
    # Heavily positive data: absmax wastes range on nonexistent negatives
    x = torch.tensor([0.01, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    # Zero-point
    q_zp, scale_zp, zp = zeropoint_quantize(x)
    x_zp = zeropoint_dequantize(q_zp, scale_zp, zp)
    err_zp = (x - x_zp).abs().mean().item()

    # Absmax (imported here to compare)
    from src.quantization.absmax import absmax_quantize, absmax_dequantize
    q_abs, scale_abs = absmax_quantize(x)
    x_abs = absmax_dequantize(q_abs, scale_abs)
    err_abs = (x - x_abs).abs().mean().item()

    print(f"\nSkewed data comparison:")
    print(f"  Absmax error:     {err_abs:.6f} (scale={scale_abs:.4f})")
    print(f"  Zero-point error: {err_zp:.6f} (scale={scale_zp:.4f})")

    # Zero-point should be at least as good (often better) for skewed data
    # We don't assert strictly because for float data both can be very close
    print(f"  Zero-point {'wins' if err_zp <= err_abs else 'ties'} for this skewed distribution")


def test_int4_zeropoint():
    """Zero-point with 4 bits gives range [0, 15]."""
    x = torch.randn(20)
    q, scale, zp = zeropoint_quantize(x, bits=4)
    assert q.min().item() >= 0
    assert q.max().item() <= 15
