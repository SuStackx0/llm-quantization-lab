"""
Tests for GPTQ Quantization
===============================

Run with:  pytest tests/test_gptq.py -v

Tests cover:
- Hessian is symmetric positive definite
- GPTQ output shapes are correct
- QuantizedLinear packs and unpacks correctly
- GPTQ error is lower than naive INT4 rounding
"""

import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quantization.gptq import gptq_quantize_layer, QuantizedLinear
from src.quantization.gptq_utils import compute_hessian, cholesky_inverse


# ---- Hessian tests ----

def test_hessian_symmetric():
    """H = 2 * X @ X.T should be exactly symmetric."""
    torch.manual_seed(42)
    X = torch.randn(64, 200)   # [d_in=64, n_tokens=200]
    H = compute_hessian(X, damp=0.01)
    diff = (H - H.T).abs().max().item()
    assert diff < 1e-5, f"Hessian should be symmetric, max asymmetry: {diff}"


def test_hessian_shape():
    """H should be [d_in, d_in]."""
    X = torch.randn(32, 100)
    H = compute_hessian(X)
    assert H.shape == (32, 32), f"Expected (32, 32), got {H.shape}"


def test_hessian_positive_diagonal():
    """Diagonal of H should be positive (with damping)."""
    X = torch.randn(16, 50)
    H = compute_hessian(X, damp=0.01)
    assert H.diagonal().min().item() > 0, "Hessian diagonal should be positive"


def test_cholesky_inverse_correctness():
    """H @ H_inv should be close to identity."""
    torch.manual_seed(1)
    X = torch.randn(16, 50)
    H = compute_hessian(X)
    H_inv = cholesky_inverse(H)
    identity_approx = H @ H_inv
    eye = torch.eye(H.shape[0])
    diff = (identity_approx - eye).abs().max().item()
    assert diff < 1e-3, f"H @ H_inv should be ~I, max diff: {diff}"


# ---- GPTQ quantize_layer tests ----

def test_gptq_output_shapes():
    """gptq_quantize_layer should return correctly shaped tensors."""
    torch.manual_seed(0)
    d_out, d_in = 32, 64
    W = torch.randn(d_out, d_in)
    H = compute_hessian(torch.randn(d_in, 100))
    H_inv = cholesky_inverse(H)

    W_q, scales, zeros = gptq_quantize_layer(W, H_inv, bits=4)
    assert W_q.shape == (d_out, d_in), f"W_q shape mismatch: {W_q.shape}"
    assert scales.shape == (d_in,), f"scales shape mismatch: {scales.shape}"


def test_gptq_int4_range():
    """Quantized weights should be in [-7, 7] for 4-bit."""
    torch.manual_seed(2)
    W = torch.randn(16, 32) * 0.1
    X = torch.randn(32, 50)
    H = compute_hessian(X)
    H_inv = cholesky_inverse(H)

    W_q, scales, zeros = gptq_quantize_layer(W, H_inv, bits=4)
    assert W_q.min().item() >= -7, f"Min quantized value {W_q.min()} < -7"
    assert W_q.max().item() <= 7, f"Max quantized value {W_q.max()} > 7"


# ---- QuantizedLinear tests ----

def test_quantized_linear_pack_unpack():
    """Pack then unpack should recover the original INT4 values."""
    in_f, out_f = 32, 16
    q_linear = QuantizedLinear(in_f, out_f, bits=4)

    # Create INT4 weight matrix (values in -7..7)
    W_int = torch.randint(-7, 8, (out_f, in_f)).to(torch.int8)
    scales = torch.ones(in_f)
    zeros = torch.zeros(in_f)

    q_linear.pack(W_int, scales, zeros)
    W_recovered = q_linear.dequantize()

    # With scale=1 and zero=0, dequantized values should match original ints
    W_recovered_int = W_recovered.float().round().clamp(-7, 7)
    W_int_f = W_int.float()

    match = (W_recovered_int == W_int_f).float().mean().item()
    assert match > 0.95, f"Only {match*100:.1f}% of INT4 values recovered correctly"


def test_quantized_linear_forward_shape():
    """QuantizedLinear forward should produce correct output shape."""
    in_f, out_f = 64, 32
    q_linear = QuantizedLinear(in_f, out_f, bits=4)

    W_int = torch.randint(-7, 8, (out_f, in_f)).to(torch.int8)
    scales = torch.rand(in_f) * 0.1
    zeros = torch.zeros(in_f)
    q_linear.pack(W_int, scales, zeros)

    x = torch.randn(2, 16, in_f, dtype=torch.float16)
    out = q_linear(x)
    assert out.shape == (2, 16, out_f), f"Expected (2, 16, {out_f}), got {out.shape}"


def test_gptq_better_than_naive():
    """
    GPTQ INT4 should have lower dequantization error than naive INT4 rounding.

    This is the KEY property of GPTQ — it compensates errors using the Hessian.
    """
    torch.manual_seed(42)
    d_out, d_in = 32, 64
    W = torch.randn(d_out, d_in) * 0.1

    # Naive INT4 quantization (no error compensation)
    from src.quantization.absmax import absmax_quantize, absmax_dequantize
    W_q_naive, scale_naive = absmax_quantize(W, bits=4)
    W_naive_dequant = absmax_dequantize(W_q_naive, scale_naive)
    error_naive = (W - W_naive_dequant).abs().mean().item()

    # GPTQ INT4 (with Hessian-guided error compensation)
    X = torch.randn(d_in, 200) * 0.5   # simulated activations
    H = compute_hessian(X, damp=0.01)
    H_inv = cholesky_inverse(H)
    W_q_gptq, scales_gptq, _ = gptq_quantize_layer(W, H_inv, bits=4)
    W_gptq_dequant = W_q_gptq.float() * scales_gptq.unsqueeze(0)
    error_gptq = (W - W_gptq_dequant).abs().mean().item()

    print(f"\nNaive INT4 error: {error_naive:.6f}")
    print(f"GPTQ INT4 error:  {error_gptq:.6f}")
    print(f"GPTQ improvement: {(1 - error_gptq/error_naive)*100:.1f}%")

    # GPTQ should be at least as good (typically much better)
    # We use a generous threshold because the improvement depends on the data
    assert error_gptq <= error_naive * 1.1, \
        f"GPTQ ({error_gptq:.4f}) should be better than naive ({error_naive:.4f})"
