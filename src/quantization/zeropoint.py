"""
Module 1B: Zero-Point (Asymmetric) Quantization
================================================

Unlike absmax (which is symmetric around 0), zero-point quantization
maps the actual [min, max] range of data to [0, 255] for UINT8.

Why asymmetric? Real weight distributions are often NOT centered at zero.
If weights range from -0.1 to 2.5, absmax wastes half the integer range
on negative numbers that barely exist. Zero-point fixes this.

Visual intuition:
  Original:  [-0.1, 0.5, 1.0, 2.0, 2.5]   actual range: [-0.1, 2.5]
  Scale:      (2.5 - (-0.1)) / 255 = 0.0102
  Zero-pt:    round(0.1 / 0.0102) = 10       (maps -0.1 → 0)
  Quantized: [0,    59,  108, 206, 255]      (uses full 0..255 range)
  Dequant:   [-0.1, 0.5, 1.0, 2.0, 2.5]    (approximately recovered)

Read docs/theory.md for a deeper explanation.
"""

import torch


def zeropoint_quantize(tensor: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, float, int]:
    """
    Quantize a float tensor using zero-point (asymmetric) scheme.

    Args:
        tensor: Any float tensor (e.g. a weight matrix)
        bits:   Bit width. 8 bits → 256 levels (0 to 255)

    Returns:
        quantized:  Unsigned integer tensor (torch.uint8), same shape
        scale:      The scale factor (size of one integer step)
        zero_point: The integer that represents 0.0 in the original data
    """
    # Step 1: find the actual range of this tensor
    min_val = tensor.min().item()
    max_val = tensor.max().item()

    # Step 2: how many integer levels do we have?
    n_levels = (2 ** bits) - 1   # 255 for 8-bit

    # Step 3: compute scale — maps the entire float range to integer range
    scale = (max_val - min_val) / n_levels

    # Avoid division by zero
    if scale == 0:
        scale = 1.0

    # Step 4: compute zero_point — the integer value that represents 0.0
    # Derived from: 0.0 = scale * (zp - zero_point) → zero_point = -min / scale
    # zero_point is the integer value that represents x=0.0 in the quantized space.
    # It can be negative (for all-positive data) or > n_levels (for all-negative data).
    # We do NOT clamp it — the quantized VALUES are clamped, but zero_point is just a shift parameter.
    zero_point = int(round(-min_val / scale))

    # Step 5: quantize — shift by zero_point then scale
    quantized = torch.round(tensor / scale + zero_point)
    quantized = quantized.clamp(0, n_levels)
    quantized = quantized.to(torch.uint8)

    return quantized, scale, zero_point


def zeropoint_dequantize(quantized: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """
    Recover approximate float values from zero-point quantized integers.

    Args:
        quantized:  uint8 tensor from zeropoint_quantize
        scale:      scale factor from zeropoint_quantize
        zero_point: zero_point from zeropoint_quantize

    Returns:
        Approximate float32 tensor
    """
    # Reverse the operation: subtract zero_point, then multiply by scale
    return (quantized.float() - zero_point) * scale


def zeropoint_quantize_per_channel(tensor: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-channel zero-point quantization — each row gets its own scale and zero_point.

    Args:
        tensor: 2D weight matrix [out_features, in_features]
        bits:   Bit width

    Returns:
        quantized:    uint8 tensor [out_features, in_features]
        scales:       [out_features, 1]
        zero_points:  [out_features, 1] (as int32 for safe arithmetic)
    """
    n_levels = (2 ** bits) - 1

    # Per-row min and max
    min_vals = tensor.min(dim=1, keepdim=True).values
    max_vals = tensor.max(dim=1, keepdim=True).values

    scales = ((max_vals - min_vals) / n_levels).clamp(min=1e-8)
    zero_points = torch.round(-min_vals / scales).clamp(0, n_levels).to(torch.int32)

    quantized = torch.round(tensor / scales + zero_points.float()).clamp(0, n_levels).to(torch.uint8)
    return quantized, scales, zero_points
