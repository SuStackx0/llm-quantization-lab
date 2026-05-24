"""
Module 2A: GPTQ Helper Utilities
==================================

These functions support the GPTQ algorithm by:
1. Collecting activation statistics from a layer during a calibration forward pass
2. Computing the Hessian matrix (tells us which weights matter most)
3. Inverting the Hessian (needed for error compensation)

Why does GPTQ need all this?
-----------------------------
Naive quantization just rounds each weight independently. But weights
don't work independently — they interact through matrix multiplication.
Quantizing weight W[i] introduces error e[i]. This error propagates
to the output and affects all subsequent computations.

The Hessian H = 2 * X @ X.T captures how the output LOSS changes
with respect to each weight. High H[i,i] means weight i is very
important (small changes → big output changes). Low H[i,i] means
weight i is less important (we can afford more error there).

GPTQ uses H_inv to "compensate" remaining weights after quantizing each one.
"""

import torch
import torch.nn as nn
from typing import Optional


def collect_input_stats(
    model: nn.Module,
    calibration_tokens: torch.Tensor,
    layer_name: str,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Run calibration data through the model and capture what a specific
    layer RECEIVES as input (its activations).

    Think of it like putting a sensor before a layer to record what
    data flows through it during real inference.

    Args:
        model:               The language model (e.g. TinyLlama)
        calibration_tokens:  Tokenized text, shape [n_samples, seq_len]
        layer_name:          Which layer to monitor (e.g. "model.layers.0.self_attn.q_proj")
        device:              "cpu" or "mps" or "cuda"

    Returns:
        X: input activations, shape [d_in, n_tokens]
           (transposed because H = 2 * X @ X.T needs this shape)
    """
    collected = []

    # A "hook" is a function that PyTorch calls automatically every time
    # a layer runs its forward pass. We use it to spy on the inputs.
    def capture_input(module, input, output):
        # input is a tuple; input[0] is the tensor flowing into the layer
        x = input[0].detach()  # detach so we don't accidentally affect gradients
        # x shape: [batch, seq_len, d_in]
        # We want [d_in, n_tokens], so reshape and transpose
        x = x.reshape(-1, x.shape[-1])   # [batch*seq_len, d_in]
        collected.append(x.T.cpu())       # [d_in, batch*seq_len]

    # Find the target layer by its name (dot-separated path through the model)
    target_layer = dict(model.named_modules())[layer_name]
    hook = target_layer.register_forward_hook(capture_input)

    # Run each calibration sample through the model
    model.eval()
    with torch.no_grad():
        for i in range(calibration_tokens.shape[0]):
            tokens = calibration_tokens[i].unsqueeze(0).to(device)
            try:
                model(tokens)
            except Exception:
                pass  # model might error on some samples — that's OK

    hook.remove()  # always clean up hooks!

    if not collected:
        raise RuntimeError(f"No activations collected for layer '{layer_name}'. Check the name.")

    # Concatenate all batches along the token dimension
    X = torch.cat(collected, dim=1)   # [d_in, total_tokens]
    return X.to(device)


def compute_hessian(X: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """
    Compute the Hessian of the layer's squared output error with respect to its weights.

    H = (2 / n_tokens) * X @ X.T

    The Hessian tells us: "if I change weight w_i by a tiny amount,
    how much does the layer's output change?" High H[i,i] = important weight.

    Args:
        X:    Input activations, shape [d_in, n_tokens]
        damp: Small value added to diagonal for numerical stability.
              Without damping, near-zero eigenvalues cause H_inv to explode.

    Returns:
        H: Hessian matrix, shape [d_in, d_in]
    """
    # Cast to float32 on CPU before the matrix multiply — fp16 loses too much
    # precision here and causes Cholesky to fail downstream.
    X = X.float().cpu()
    n_tokens = X.shape[1]

    # Core formula: H = 2 * X @ X.T / n_tokens
    H = (2.0 / n_tokens) * (X @ X.T)

    # Damping: add a small fraction of the mean diagonal to the diagonal.
    # This prevents singular matrices and improves numerical stability.
    # Think of it as saying "every weight matters at least a little bit."
    mean_diag = H.diagonal().mean().item()
    H += damp * mean_diag * torch.eye(H.shape[0], dtype=torch.float32)

    return H


def cholesky_inverse(H: torch.Tensor) -> torch.Tensor:
    """
    Compute H_inv (inverse of the Hessian) using Cholesky decomposition.

    Why not just torch.inverse(H)?
    Direct inversion is numerically unstable for large matrices.
    Cholesky decomposes H = L @ L.T (since H is symmetric positive definite),
    then inverses it stably. This is the standard approach in GPTQ.

    Args:
        H: Symmetric positive definite matrix [d_in, d_in]

    Returns:
        H_inv: Inverse of H, same shape [d_in, d_in]
    """
    # cholesky_inverse is not implemented on MPS; always compute on CPU
    H_cpu = H.float().cpu()

    # Retry with increasing diagonal damping before giving up on Cholesky
    for extra_damp in [0.0, 1e-3, 1e-2, 1e-1]:
        try:
            H_damped = H_cpu + extra_damp * torch.eye(H_cpu.shape[0])
            L = torch.linalg.cholesky(H_damped)
            H_inv = torch.cholesky_inverse(L)
            return H_inv.to(dtype=H.dtype, device=H.device)
        except torch.linalg.LinAlgError:
            continue

    # Last resort: pseudoinverse with small regularization to avoid SVD blow-up
    print("  Warning: Cholesky failed, using pseudoinverse instead")
    H_reg = H_cpu + 1e-6 * torch.eye(H_cpu.shape[0])
    H_inv = torch.linalg.pinv(H_reg)
    return H_inv.to(dtype=H.dtype, device=H.device)
