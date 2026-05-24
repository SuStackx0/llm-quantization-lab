"""
Module 3B: K-Quant Dequantization
===================================

K-quants are llama.cpp's clever quantization format that uses
MIXED PRECISION within each "super-block" of weights.

Why k-quants beat uniform quantization:
-----------------------------------------
Uniform INT4: every weight uses 4 bits, global scale per tensor
K-quants:     every BLOCK of 32 weights gets its OWN scale
              blocks are grouped into SUPER-BLOCKS of 256 weights
              super-block stores scales using 6-bit precision

This hierarchical scale storage is the key insight.
More bits for scales → less quantization error per block.

Block layouts implemented here:

  Q8_0 (simple, good starting point):
    [2 bytes: d (fp16 scale)] [32 bytes: 32 × int8 values]
    Total: 34 bytes per 32 weights = 8.5 bits/weight
    Dequant: w[i] = d * q[i]

  Q4_K (the most common k-quant format):
    [2 bytes: d (fp16 super-scale)]
    [2 bytes: dmin (fp16 super-min)]
    [12 bytes: 6-bit scales and mins for 8 sub-blocks]
    [128 bytes: 256 × 4-bit quantized values (2 per byte)]
    Total: 144 bytes per 256 weights = 4.5 bits/weight
    Dequant: w[i] = scale[sub] * q[i] - min[sub]

  Q5_K (similar to Q4_K but with 5 bits):
    Same layout + 32 bytes of "high bits" for extra precision
    Total: 176 bytes per 256 weights = 5.5 bits/weight
"""

import torch
import numpy as np
import struct


def dequantize_q8_0(data: bytes, shape: tuple) -> torch.Tensor:
    """
    Dequantize Q8_0 format: 8-bit symmetric per 32-weight block.

    Block layout (34 bytes total per block):
      - 2 bytes: d (scale, stored as float16)
      - 32 bytes: 32 signed int8 quantized values

    Dequantize: weight[i] = d * q[i]

    This is the simplest k-quant — a good format to understand first.

    Args:
        data:  Raw bytes from the GGUF file for this tensor
        shape: Target output shape (e.g. (4096, 4096))

    Returns:
        float32 tensor of shape `shape`
    """
    BLOCK_SIZE = 32
    BYTES_PER_BLOCK = 34   # 2 (scale) + 32 (values)

    n_elements = 1
    for d in shape:
        n_elements *= d

    n_blocks = n_elements // BLOCK_SIZE
    assert len(data) >= n_blocks * BYTES_PER_BLOCK, \
        f"Data too short: need {n_blocks * BYTES_PER_BLOCK} bytes, got {len(data)}"

    weights = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        block_start = i * BYTES_PER_BLOCK
        block = data[block_start: block_start + BYTES_PER_BLOCK]

        # Read float16 scale (first 2 bytes)
        d = struct.unpack_from("<e", block, 0)[0]   # "e" = float16

        # Read 32 int8 values (bytes 2..33)
        qs = np.frombuffer(block, dtype=np.int8, count=32, offset=2)

        # Dequantize
        elem_start = i * BLOCK_SIZE
        weights[elem_start: elem_start + BLOCK_SIZE] = d * qs.astype(np.float32)

    return torch.from_numpy(weights).reshape(shape)


def dequantize_q4_k(data: bytes, shape: tuple) -> torch.Tensor:
    """
    Dequantize Q4_K format: 4-bit with mixed 6-bit scales per 256-weight super-block.

    This is the "M" (medium) variant used in Q4_K_M — the most popular GGUF format
    because it offers the best quality/size tradeoff for most use cases.

    Super-block layout (144 bytes total per 256 weights):
      - Bytes 0-1:    d     (super-block scale, fp16)
      - Bytes 2-3:    dmin  (super-block min, fp16)
      - Bytes 4-15:   12 bytes encoding 6-bit sub-scales and sub-mins
                      for 8 sub-blocks of 32 weights each
      - Bytes 16-143: 128 bytes = 256 × 4-bit quantized values (2 per byte)

    For each sub-block i (32 weights):
      scale = d * sub_scale[i]
      min   = dmin * sub_min[i]
      weight[j] = scale * q[j] - min

    The 6-bit scale encoding is the tricky part — see _decode_scales_6bit().

    Args:
        data:  Raw bytes for this tensor
        shape: Target shape

    Returns:
        float32 tensor
    """
    BLOCK_SIZE = 256
    BYTES_PER_BLOCK = 144   # 2 + 2 + 12 + 128

    n_elements = 1
    for d in shape:
        n_elements *= d

    n_blocks = n_elements // BLOCK_SIZE
    if n_blocks == 0:
        # For small tensors with fewer than 256 elements, fall back to Q8_0
        return dequantize_q8_0(data, shape)

    weights = np.zeros(n_elements, dtype=np.float32)

    for blk in range(n_blocks):
        blk_start = blk * BYTES_PER_BLOCK
        block = data[blk_start: blk_start + BYTES_PER_BLOCK]

        # Read super-block scale and min
        d = struct.unpack_from("<e", block, 0)[0]      # float16
        dmin = struct.unpack_from("<e", block, 2)[0]   # float16

        # Decode 8 sub-block scales and mins from 12 packed bytes
        sub_scales, sub_mins = _decode_scales_6bit(block[4:16])

        # Read 256 × 4-bit quantized values from bytes 16..143
        qs = np.frombuffer(block, dtype=np.uint8, count=128, offset=16)

        # Unpack 4-bit values: low nibble and high nibble of each byte
        q_low  = qs & 0x0F          # lower 4 bits of each byte → first 128 values
        q_high = (qs >> 4) & 0x0F   # upper 4 bits of each byte → second 128 values
        q_all  = np.concatenate([q_low, q_high])   # shape [256]

        # Dequantize each sub-block (8 sub-blocks × 32 weights = 256 total)
        elem_start = blk * BLOCK_SIZE
        for sub in range(8):
            scale = d * sub_scales[sub]
            mn    = dmin * sub_mins[sub]
            sub_start = sub * 32
            sub_q = q_all[sub_start: sub_start + 32].astype(np.float32)
            weights[elem_start + sub_start: elem_start + sub_start + 32] = scale * sub_q - mn

    return torch.from_numpy(weights).reshape(shape)


def dequantize_q5_k(data: bytes, shape: tuple) -> torch.Tensor:
    """
    Dequantize Q5_K format: 5-bit with mixed 6-bit scales.

    Same structure as Q4_K but each value has an extra "high bit" stored
    in a separate 32-byte array, giving 5 bits total per weight.

    Super-block layout (176 bytes total per 256 weights):
      - Bytes 0-1:    d     (fp16 super-scale)
      - Bytes 2-3:    dmin  (fp16 super-min)
      - Bytes 4-15:   12 bytes of 6-bit sub-scales and sub-mins
      - Bytes 16-47:  32 bytes of high bits (1 extra bit per weight)
      - Bytes 48-175: 128 bytes of low 4 bits (same as Q4_K)

    Dequant: weight[j] = scale * (q_low[j] | (q_high_bit[j] << 4)) - min
    """
    BLOCK_SIZE = 256
    BYTES_PER_BLOCK = 176

    n_elements = 1
    for d in shape:
        n_elements *= d

    n_blocks = n_elements // BLOCK_SIZE
    if n_blocks == 0:
        return dequantize_q8_0(data, shape)

    weights = np.zeros(n_elements, dtype=np.float32)

    for blk in range(n_blocks):
        blk_start = blk * BYTES_PER_BLOCK
        block = data[blk_start: blk_start + BYTES_PER_BLOCK]

        d    = struct.unpack_from("<e", block, 0)[0]
        dmin = struct.unpack_from("<e", block, 2)[0]

        sub_scales, sub_mins = _decode_scales_6bit(block[4:16])

        # 32 bytes of high bits (1 bit per weight, packed as 32 bytes × 8 bits)
        qh = np.frombuffer(block, dtype=np.uint8, count=32, offset=16)  # [32]
        # Unpack 8 high bits from each byte → shape [256]
        high_bits = np.unpackbits(qh, bitorder="little").reshape(32, 8).flatten()[:256]

        # 128 bytes of low 4 bits
        qs = np.frombuffer(block, dtype=np.uint8, count=128, offset=48)
        q_low  = qs & 0x0F
        q_high_nibble = (qs >> 4) & 0x0F
        q_low4 = np.concatenate([q_low, q_high_nibble])  # [256] low 4 bits

        # Combine: 5-bit value = q_low4 | (high_bit << 4)
        q5 = q_low4 | (high_bits << 4)   # [256], values 0..31

        elem_start = blk * BLOCK_SIZE
        for sub in range(8):
            scale = d * sub_scales[sub]
            mn    = dmin * sub_mins[sub]
            sub_start = sub * 32
            sub_q = q5[sub_start: sub_start + 32].astype(np.float32)
            weights[elem_start + sub_start: elem_start + sub_start + 32] = scale * sub_q - mn

    return torch.from_numpy(weights).reshape(shape)


def _decode_scales_6bit(scale_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode the 12 bytes that encode 8 sub-block scales and 8 sub-block mins.

    Each scale/min is stored as a 6-bit value (0..63).
    Layout: 8 scales + 8 mins = 16 values × 6 bits = 96 bits = 12 bytes.

    Bit packing (llama.cpp convention):
      For i in 0..5:  scales[i]  = byte[i] & 0x3F  (lower 6 bits)
      For i in 0..5:  mins[i]    = byte[i+6] & 0x3F
      Extra bits from byte[8..11] patch scales[6..7] and mins[6..7]

    Returns:
        sub_scales: np.ndarray of shape [8], dtype float32 (values 0..63)
        sub_mins:   np.ndarray of shape [8], dtype float32 (values 0..63)
    """
    b = np.frombuffer(scale_bytes, dtype=np.uint8)

    sub_scales = np.zeros(8, dtype=np.float32)
    sub_mins   = np.zeros(8, dtype=np.float32)

    # First 6 sub-scales from bytes 0..5 (lower 6 bits each)
    for i in range(6):
        sub_scales[i] = b[i] & 0x3F

    # First 6 sub-mins from bytes 6..11 (lower 6 bits each)
    for i in range(6):
        sub_mins[i] = b[i + 6] & 0x3F

    # Last 2 scales and mins are assembled from upper bits
    # bytes 0..3 upper 2 bits contribute to scales[6] and scales[7]
    # bytes 4..7 upper 2 bits contribute to mins[6] and mins[7]
    sub_scales[6] = (b[0] >> 6) | ((b[1] >> 6) << 2) | ((b[2] >> 6) << 4)
    sub_scales[7] = (b[3] >> 6) | ((b[4] >> 6) << 2) | ((b[5] >> 6) << 4)
    sub_mins[6]   = (b[6] >> 6) | ((b[7] >> 6) << 2) | ((b[8] >> 6) << 4)
    sub_mins[7]   = (b[9] >> 6) | ((b[10] >> 6) << 2) | ((b[11] >> 6) << 4)

    return sub_scales, sub_mins
