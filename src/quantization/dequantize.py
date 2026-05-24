"""
Shared dequantization utility.

Given a quantized tensor and its metadata (scheme, scale, zero_point),
returns the approximate float tensor.

This is used by QuantizedLinear during the forward pass — weights are
stored as integers but dequantized just-in-time before the matmul.
"""

import torch


def dequantize_tensor(
    quantized: torch.Tensor,
    scale,
    zero_point=None,
    scheme: str = "absmax",
) -> torch.Tensor:
    """
    Universal dequantization function.

    Args:
        quantized:   Integer tensor (int8 for absmax, uint8 for zeropoint)
        scale:       Scale factor (float or tensor of floats for per-channel)
        zero_point:  Only needed for zeropoint scheme (int or tensor of ints)
        scheme:      "absmax" or "zeropoint"

    Returns:
        float32 tensor, approximately equal to the original before quantization
    """
    if scheme == "absmax":
        # Absmax: just multiply by scale
        if isinstance(scale, torch.Tensor):
            # Per-channel: scales shape is [out, 1], broadcasts correctly
            return quantized.float() * scale.float()
        return quantized.float() * scale

    elif scheme == "zeropoint":
        # Zeropoint: subtract zero_point, then multiply by scale
        if isinstance(zero_point, torch.Tensor):
            return (quantized.float() - zero_point.float()) * scale.float()
        return (quantized.float() - zero_point) * scale

    else:
        raise ValueError(f"Unknown scheme '{scheme}'. Use 'absmax' or 'zeropoint'.")
