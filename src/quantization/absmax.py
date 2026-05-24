"""
Module 1A: Absmax (Symmetric) Quantization
===========================================

The simplest quantization scheme. Maps the range [-max_val, +max_val]
evenly onto [-127, 127] for INT8, or [-7, 7] for INT4.

Why "absmax"? Because the scale is based on the absolute maximum value.

Visual intuition:
  Original:  [-2.5, -1.0, 0.0, 1.0, 2.5]   (floats, lots of memory)
  Scale:      2.5 / 127 = 0.0197
  Quantized: [-127, -51,  0,   51,  127]     (integers, 4x less memory)
  Dequant:   [-2.5, -1.0, 0.0, 1.0, 2.5]   (approximately recovered)

Read docs/theory.md for a deeper explanation.
"""

import torch


def absmax_quantize(tensor: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, float]:
    """
    Quantize a float tensor to integers using absmax (symmetric) scheme.

    Args:
        tensor: Any float tensor (e.g. a weight matrix from a neural network)
        bits:   How many bits to use. 8 = INT8 (range -127..127), 4 = INT4 (range -7..7)

    Returns:
        quantized: Integer tensor (same shape, stored as int8)
        scale:     The scale factor needed to recover the original values
    """
    # Step 1: figure out the largest absolute value in this tensor
    max_val = tensor.abs().max().item()

    # Step 2: compute scale — this is the "size" of one integer step
    # For 8 bits: 2^(8-1) - 1 = 127 levels on each side of zero
    n_levels = (2 ** (bits - 1)) - 1   # 127 for INT8, 7 for INT4
    scale = max_val / n_levels

    # Avoid division by zero if the tensor is all zeros
    if scale == 0:
        scale = 1.0

    # Step 3: divide every value by the scale and round to nearest integer
    quantized = torch.round(tensor / scale)

    # Step 4: clamp to valid integer range (in case of floating point rounding)
    quantized = quantized.clamp(-n_levels, n_levels)

    # Step 5: store as int8 (1 byte per value instead of 4 bytes for float32)
    quantized = quantized.to(torch.int8)

    return quantized, scale


def absmax_dequantize(quantized: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Recover approximate float values from quantized integers.

    This is NOT lossless — some precision is permanently lost during quantization.
    The error is called "quantization error" and is analyzed in docs/theory.md.

    Args:
        quantized: Integer tensor from absmax_quantize
        scale:     The scale factor returned by absmax_quantize

    Returns:
        Approximate float32 tensor (same shape as original)
    """
    # Simply multiply each integer back by the scale
    # int8 -> float32 first, then scale
    return quantized.float() * scale


def absmax_quantize_per_channel(tensor: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel (per-row) quantization — each row of a weight matrix gets its own scale.

    Why this is better: different rows can have very different value ranges.
    A single global scale wastes precision for rows with small values.

    Args:
        tensor: 2D weight matrix [out_features, in_features]
        bits:   Bit width

    Returns:
        quantized: Integer tensor [out_features, in_features]
        scales:    One scale per row [out_features, 1]
    """
    n_levels = (2 ** (bits - 1)) - 1

    # Find max absolute value PER ROW (keepdim=True keeps shape [out, 1])
    max_vals = tensor.abs().max(dim=1, keepdim=True).values
    scales = (max_vals / n_levels).clamp(min=1e-8)

    quantized = torch.round(tensor / scales).clamp(-n_levels, n_levels).to(torch.int8)
    return quantized, scales
