"""
Module 2B: GPTQ (Post-Training Quantization)
=============================================

GPTQ is the algorithm behind AutoGPTQ, ExLlamaV2, and most INT4 GGUF models.
Paper: "GPTQ: Accurate Post-Training Quantization" (Frantar et al., 2022)

The core insight over naive INT4 quantization:
-----------------------------------------------
Naive: round each weight independently → lots of accumulated error
GPTQ:  after quantizing weight w_i, propagate the rounding error to the
       remaining weights so THEY can compensate for the mistake.

This is like a relay race: when one runner stumbles, the next runner
starts a bit further ahead to compensate for the lost ground.

The compensation amount is guided by H_inv:
  - High H_inv[i,j] → weight j can compensate a lot for error in weight i
  - Low H_inv[i,j]  → weight j doesn't interact much with weight i

Algorithm (per layer):
  for j in range(d_in):
      1. Quantize column j: W_q[:,j] = round(W[:,j] / scale)
      2. Compute error: e = W[:,j] - dequant(W_q[:,j])    ← what we lost
      3. Spread error to future columns:
         W[:,j+1:] -= outer(e, H_inv[j, j+1:]) / H_inv[j,j]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .absmax import absmax_quantize, absmax_dequantize


def gptq_quantize_layer(
    W: torch.Tensor,
    H_inv: torch.Tensor,
    bits: int = 4,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply GPTQ to quantize a single weight matrix W using the inverse Hessian.

    Args:
        W:          Weight matrix [d_out, d_in] — rows = output neurons, cols = inputs
        H_inv:      Inverse Hessian [d_in, d_in]
        bits:       Target bit width (4 for INT4)
        block_size: Process this many columns at once (memory efficiency)

    Returns:
        W_quantized: Quantized weight matrix [d_out, d_in], stored as float
                     (the integer values, not yet scaled back to float range)
        scales:      Per-column scale factors [d_in]
        zeros:       Per-column zero points [d_in] (for asymmetric within GPTQ)
    """
    d_out, d_in = W.shape
    device = W.device

    # H_inv may have been computed on CPU (cholesky not supported on MPS)
    H_inv = H_inv.to(device)

    # We'll work on a copy — GPTQ modifies W in-place as it goes
    W = W.clone().float()

    scales = torch.zeros(d_in, device=device)
    zeros = torch.zeros(d_in, device=device)
    W_quantized = torch.zeros_like(W)

    n_levels = (2 ** (bits - 1)) - 1   # 7 for INT4

    # Process column by column
    for j in range(d_in):
        col = W[:, j]  # shape [d_out] — all output neurons for input j

        # Step 1: compute scale for this column (per-column absmax)
        max_val = col.abs().max().item()
        scale = max_val / n_levels if max_val > 0 else 1.0
        scales[j] = scale

        # Step 2: quantize this column
        q = torch.round(col / scale).clamp(-n_levels, n_levels)
        W_quantized[:, j] = q

        # Step 3: compute the rounding error we just introduced
        dequantized_col = q * scale
        error = col - dequantized_col  # shape [d_out]

        # Step 4: propagate error to all REMAINING columns (j+1 onwards)
        # This is the key GPTQ step: adjust remaining weights to compensate
        if j + 1 < d_in and H_inv[j, j].abs() > 1e-8:
            # H_inv[j, j+1:] tells us how much weight k responds to error in weight j
            # error has shape [d_out], H_inv[j, j+1:] has shape [d_in-j-1]
            # outer product gives [d_out, d_in-j-1] — one correction per (output, future_input)
            correction = torch.outer(error, H_inv[j, j + 1:]) / H_inv[j, j]
            W[:, j + 1:] -= correction

    return W_quantized, scales, zeros


class QuantizedLinear(nn.Module):
    """
    A replacement for nn.Linear that stores weights in INT4 format.

    Instead of 4 bytes per weight (float32), we store ~0.5 bytes per weight
    by packing two 4-bit values into one 8-bit integer.

    During the forward pass, we:
    1. Unpack the INT4 weights
    2. Dequantize to float16
    3. Do the matmul as normal

    This is called "weight-only quantization" — activations stay in float16.
    Memory savings: ~4x vs FP32, ~2x vs FP16.
    """

    def __init__(self, in_features: int, out_features: int, bits: int = 4, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # We'll store packed weights (two INT4 per byte)
        # Shape: [out_features, in_features // 2] for 4-bit packing
        self.register_buffer("weight_packed", torch.zeros(out_features, (in_features + 1) // 2, dtype=torch.uint8))
        self.register_buffer("scales", torch.ones(in_features))
        self.register_buffer("zeros", torch.zeros(in_features))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None

    def pack(self, W_int: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
        """
        Store quantized weights and their metadata.

        W_int: integer weight matrix [out_features, in_features], values in [-7, 7]
        scales: scale per column [in_features]
        zeros: zero point per column [in_features]
        """
        self.scales.copy_(scales)
        self.zeros.copy_(zeros)

        # Pack two INT4 values into one byte:
        # For columns j=0,1 (0-indexed), we store:
        #   byte = (W[:, 0] & 0x0F) | ((W[:, 1] & 0x0F) << 4)
        # This halves the memory required
        W_int = W_int.to(torch.int8)
        d_out, d_in = W_int.shape

        # Pad to even number of columns if needed
        if d_in % 2 != 0:
            W_int = torch.cat([W_int, torch.zeros(d_out, 1, dtype=torch.int8)], dim=1)

        # Shift values from [-7,7] to [0,14] so they fit in 4 bits unsigned
        W_shifted = (W_int + 8).clamp(0, 15).to(torch.uint8)

        # Pack: low nibble = even columns, high nibble = odd columns
        W_packed = W_shifted[:, 0::2] | (W_shifted[:, 1::2] << 4)
        self.weight_packed.copy_(W_packed[:, : self.weight_packed.shape[1]])

    def dequantize(self) -> torch.Tensor:
        """Unpack INT4 weights and convert to float16."""
        d_out = self.out_features
        d_in = self.in_features

        # Unpack: separate low and high nibbles
        low = self.weight_packed & 0x0F       # even columns
        high = (self.weight_packed >> 4) & 0x0F  # odd columns

        # Interleave back: [col0, col1, col2, col3, ...]
        W_flat = torch.zeros(d_out, low.shape[1] * 2, dtype=torch.uint8, device=self.weight_packed.device)
        W_flat[:, 0::2] = low
        W_flat[:, 1::2] = high

        # Trim to original size and shift back to signed range
        W_flat = W_flat[:, :d_in].to(torch.int8) - 8   # back to [-7, 7]

        # Dequantize: multiply by per-column scales; return float32 by default
        # The caller (forward) will cast to the input's dtype
        W_fp = W_flat.float() * self.scales.float().unsqueeze(0)  # [d_out, d_in]
        return W_fp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match all dtypes to the input dtype so F.linear doesn't complain
        W = self.dequantize().to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, W, bias)

    @classmethod
    def from_float(cls, linear: nn.Linear, W_int: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, bits: int = 4):
        """Create a QuantizedLinear from an existing nn.Linear + GPTQ output."""
        q_linear = cls(linear.in_features, linear.out_features, bits=bits, bias=linear.bias is not None)
        q_linear.pack(W_int, scales, zeros)
        if linear.bias is not None:
            q_linear.bias.copy_(linear.bias.data)
        return q_linear
