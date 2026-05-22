# Theory: How LLM Quantization Works

> "Quantization is the art of representing high-precision numbers with fewer bits while losing as little information as possible."

This document explains the mathematics behind every technique in this project. Read it alongside the code in `src/quantization/`.

---

## Table of Contents

1. [Why Quantization?](#1-why-quantization)
2. [The Basic Math: What Is Quantization?](#2-the-basic-math-what-is-quantization)
3. [Module 1A: Absmax (Symmetric) Quantization](#3-module-1a-absmax-symmetric-quantization)
4. [Module 1B: Zero-Point (Asymmetric) Quantization](#4-module-1b-zero-point-asymmetric-quantization)
5. [Per-Tensor vs Per-Channel Quantization](#5-per-tensor-vs-per-channel-quantization)
6. [Quantization Error — Measuring What We Lose](#6-quantization-error--measuring-what-we-lose)
7. [Module 2: GPTQ — Smart INT4 Quantization](#7-module-2-gptq--smart-int4-quantization)
8. [Module 3: K-Quants — Mixed Precision Blocks](#8-module-3-k-quants--mixed-precision-blocks)
9. [The Memory/Quality Tradeoff Curve](#9-the-memoryquality-tradeoff-curve)
10. [Weight-Only vs Activation Quantization](#10-weight-only-vs-activation-quantization)

---

## 1. Why Quantization?

**The problem:** Modern LLMs are huge.

| Model | Parameters | FP32 Memory |
|-------|-----------|-------------|
| TinyLlama-1.1B | 1.1 billion | 4.4 GB |
| LLaMA-3.1-8B | 8 billion | 32 GB |
| LLaMA-3.1-70B | 70 billion | 280 GB |

FP32 (float32) stores each number in **4 bytes**. A 70B model needs 4 × 70,000,000,000 = **280 GB** of RAM. That's 3-4 high-end GPUs just to load the model.

**The solution:** Store numbers in fewer bits.

```
FP32:  4 bytes per weight → 280 GB for 70B model
FP16:  2 bytes per weight → 140 GB
INT8:  1 byte per weight  → 70 GB
INT4:  0.5 bytes per weight → 35 GB  ← fits on one A100!
```

The goal: get as close to INT4 as possible while keeping the model useful.

---

## 2. The Basic Math: What Is Quantization?

Quantization maps a continuous range of float values to a discrete set of integers.

```
Float space:    -2.5 ... -1.0 ... 0.0 ... 1.0 ... 2.5
                 |         |       |       |       |
INT8 space:    -127      -51      0       51      127
```

The mapping is defined by a **scale factor** (and optionally a **zero-point**):

```
quantize:   q = round(x / scale)
dequantize: x̂ = q × scale
```

The recovered value `x̂` is NOT the same as the original `x`. The difference `x - x̂` is the **quantization error**. Our goal: minimize this error.

---

## 3. Module 1A: Absmax (Symmetric) Quantization

**Code:** `src/quantization/absmax.py`

### The Formula

```
scale = max(|tensor|) / (2^(bits-1) - 1)

For INT8: scale = max(|W|) / 127
For INT4: scale = max(|W|) / 7

quantized = round(W / scale).clamp(-127, 127)
dequantized = quantized × scale
```

### Visual Example

Suppose our weight tensor has values: `[-2.4, -1.0, 0.0, 0.8, 2.0]`

```
max absolute value = 2.4
scale = 2.4 / 127 = 0.0189

Quantize:
  -2.4 / 0.0189 = -127  → round(-127) = -127
  -1.0 / 0.0189 = -52.9 → round(-53)  = -53
   0.0 / 0.0189 =  0.0  → round(0)    =   0
   0.8 / 0.0189 =  42.3 → round(42)   =  42
   2.0 / 0.0189 = 105.8 → round(106)  = 106

Integer tensor: [-127, -53, 0, 42, 106]

Dequantize:
  -127 × 0.0189 = -2.40   ✓ (exact because it was max)
   -53 × 0.0189 = -1.002  ≈ -1.0  (small error)
     0 × 0.0189 =  0.00   ✓
    42 × 0.0189 =  0.794  ≈ 0.8   (small error)
   106 × 0.0189 =  2.003  ≈ 2.0   (small error)
```

### Why "Symmetric"?

The integer range `[-127, 127]` is symmetric around 0. The float range `[-max, max]` is also symmetric around 0. This means:

- If `x = 0.0`, then `quantized = 0` (zero is perfectly representable)
- The positive and negative ranges get equal treatment

### When Absmax Fails

If the data is skewed — e.g., all values are between 0.5 and 3.0 — absmax wastes half the integer range on negative numbers that don't exist. See zero-point quantization for a fix.

---

## 4. Module 1B: Zero-Point (Asymmetric) Quantization

**Code:** `src/quantization/zeropoint.py`

### The Problem Absmax Misses

Consider weights in range `[0.5, 3.0]` (ReLU activations are always positive).

Absmax: `scale = 3.0 / 127 = 0.0236`. The negative side `[-3.0, 0.0]` is wasted.

Zero-point maps `[min, max]` directly to `[0, 255]`. **No wasted range.**

### The Formula

```
scale      = (max - min) / (2^bits - 1)    # for INT8: scale = (max - min) / 255
zero_point = round(-min / scale)            # integer that represents 0.0

quantize:   q = round(x / scale + zero_point).clamp(0, 255)
dequantize: x̂ = (q - zero_point) × scale
```

### Visual Example

Weights: `[0.5, 1.0, 2.0, 2.5, 3.0]`, range `[0.5, 3.0]`

```
scale      = (3.0 - 0.5) / 255 = 0.0098
zero_point = round(-0.5 / 0.0098) = round(-51) = -51
           → actually clamp to 0: zero_point = 0
           (since min=0.5 > 0, zero_point is negative but we clamp)

With proper formula:
  zero_point = round(-min / scale) = round(-0.5 / 0.0098) = round(-51) = 0 after clamp

Let me redo with min = -0.5 to show it better:
  Weights: [-0.5, 0.0, 1.0, 2.0, 3.0]
  scale = (3.0 - (-0.5)) / 255 = 0.0137
  zero_point = round(0.5 / 0.0137) = round(36.5) = 37

  Quantize:
    -0.5 → round(-0.5 / 0.0137 + 37) = round(0) = 0   ← maps min to 0
     0.0 → round(0 / 0.0137 + 37) = 37                 ← 0.0 maps to zero_point=37
     3.0 → round(3.0 / 0.0137 + 37) = round(256) ≈ 255 ← maps max to 255

  Dequantize:
     0 → (0 - 37) × 0.0137 = -0.507 ≈ -0.5   ✓
    37 → (37 - 37) × 0.0137 = 0.0             ✓
   255 → (255 - 37) × 0.0137 = 2.987 ≈ 3.0   ✓
```

### When to Use Zero-Point vs Absmax

| Situation | Better Choice |
|-----------|--------------|
| Weights (usually near-symmetric) | Either works well |
| Activations after ReLU (always ≥ 0) | Zero-point (uses full range) |
| Activations with mixed positive/negative | Absmax (simpler) |
| Memory is critical | Zero-point (more efficient for skewed data) |

---

## 5. Per-Tensor vs Per-Channel Quantization

So far we've used one scale for the entire weight matrix. This is called **per-tensor** quantization.

**The problem:** Different rows of a weight matrix can have wildly different value ranges.

```
Row 0: [-0.01, 0.02, -0.01, 0.03]   → max = 0.03
Row 1: [-5.0,  2.0,  -3.0,  4.0]   → max = 5.0
Row 2: [0.001, 0.002, 0.001, 0.002] → max = 0.002
```

If we use one scale for the whole matrix: `scale = 5.0 / 127 = 0.0394`

Row 0 quantizes to:  `round(0.03 / 0.0394) = 1`  — only 1 integer level! Terrible.
Row 2 quantizes to:  `round(0.002 / 0.0394) = 0` — everything rounds to 0. Catastrophic.

**Per-channel (per-row) quantization** gives each row its own scale:

```
Row 0: scale = 0.03 / 127 = 0.000236  → uses 127 levels
Row 1: scale = 5.0 / 127  = 0.0394   → uses 127 levels
Row 2: scale = 0.002 / 127 = 0.0000157 → uses 127 levels ← 
```

All rows now use the full precision available. Per-channel quantization is almost always better for weight matrices.

**Code:** `absmax_quantize_per_channel()` in `src/quantization/absmax.py`

---

## 6. Quantization Error — Measuring What We Lose

**Quantization error** = the difference between original and recovered values:

```python
error = dequantize(quantize(x)) - x
```

For a uniform quantizer (like absmax), the maximum error is `scale / 2`.

This is because rounding always introduces at most half a step of error.

**Why INT4 is harder than INT8:**

```
INT8: 127 levels each side of 0  → fine-grained steps
INT4: 7 levels each side of 0   → coarse steps

For a weight in [0, 1.0]:
  INT8: scale = 1/127 = 0.0079, max error = 0.004   (0.4% of range)
  INT4: scale = 1/7   = 0.143,  max error = 0.071   (7.1% of range)
```

INT4 error is 18× larger than INT8. This is why INT4 requires GPTQ — naive rounding causes too much degradation.

**How to visualize quantization error:**

```python
import matplotlib.pyplot as plt
import torch
from src.quantization.absmax import absmax_quantize, absmax_dequantize

W = torch.randn(256, 256) * 0.1
q_int8, s8 = absmax_quantize(W, bits=8)
q_int4, s4 = absmax_quantize(W, bits=4)

err_int8 = (W - absmax_dequantize(q_int8, s8)).abs()
err_int4 = (W - absmax_dequantize(q_int4, s4)).abs()

plt.figure(figsize=(12, 4))
plt.subplot(1,3,1); plt.hist(W.flatten().numpy(), 50); plt.title("Original")
plt.subplot(1,3,2); plt.imshow(err_int8.numpy()); plt.title("INT8 Error"); plt.colorbar()
plt.subplot(1,3,3); plt.imshow(err_int4.numpy()); plt.title("INT4 Error"); plt.colorbar()
plt.show()
```

---

## 7. Module 2: GPTQ — Smart INT4 Quantization

**Code:** `src/quantization/gptq.py`, `src/quantization/gptq_utils.py`

**Paper:** "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2022)

### The Problem with Naive INT4

When we naively round weights to INT4:
- Each weight's rounding error is independent
- Errors accumulate across the entire layer
- The output distribution shifts significantly
- Perplexity degrades from ~7 to ~15+ for a 1B model

### The GPTQ Insight

Quantization error at the **output** of a layer depends on:
1. The rounding error of each weight
2. How "important" that weight is (how much the output changes when it changes)

**Key insight:** We can measure weight importance using the Hessian matrix.

### The Hessian Matrix

For a linear layer `y = Wx`:

The Hessian `H = 2 * X @ X.T / n`

where `X` is the input activation matrix collected during a calibration forward pass.

**Meaning of H[i,i]:**
- High H[i,i] → weight i has a large effect on output → quantize it carefully
- Low H[i,i]  → weight i barely affects output → we can afford more error here

### The GPTQ Algorithm

For each column j (one at a time, left to right):

```
Step 1: Quantize w_j
  w_q[j] = round(w[j] / scale)

Step 2: Compute rounding error
  e[j] = w[j] - dequant(w_q[j])

Step 3: Compensate remaining weights
  w[j+1:] -= outer(e, H_inv[j, j+1:]) / H_inv[j, j]
```

The key formula `w[j+1:] -= outer(e, H_inv[j, j+1:]) / H_inv[j, j]` says:

> "I just introduced error `e` when quantizing weight j. Based on how weight j interacts with future weights (from H_inv), I'll adjust all future weights to partially cancel this error."

This is a **greedy algorithm** — it can't undo past mistakes but it prevents them from compounding.

### Why Cholesky for H_inv?

The Hessian H can be nearly singular (some eigenvalues close to 0) for real language models. Direct inversion `torch.inverse(H)` amplifies tiny numerical errors into huge inversion errors.

Cholesky decomposition: `H = L @ L.T` (L is lower triangular)

Then: `H_inv = (L.T)^{-1} @ L^{-1}`

Triangular inversion is numerically stable. This is why `cholesky_inverse()` in `gptq_utils.py` uses it instead of direct inversion.

### Memory Trick: Block Size

Processing one column at a time would require `d_in` operations (e.g., 4096 for LLaMA). For efficiency, GPTQ processes `block_size=128` columns at once. This reduces memory bandwidth while maintaining the same mathematical result.

---

## 8. Module 3: K-Quants — Mixed Precision Blocks

**Code:** `src/gguf/k_quants.py`

K-quants are llama.cpp's format, used by Ollama and most local LLM deployments.

### The Problem with Uniform Block Quantization

Simple block quantization (like Q4_0):
- Divide weights into blocks of 32
- Each block gets one scale: `scale = max(|block|) / 7`
- All 32 weights use the same scale

Problem: if one block has a single outlier weight of magnitude 10, the scale becomes `10/7 = 1.43`. All other weights in the block (say, magnitude 0.1) get rounded to 0 (since `0.1/1.43 = 0.07 → round to 0`).

### The K-Quant Solution: Hierarchical Scales

K-quants organize weights into **super-blocks** of 256 weights, each subdivided into **8 sub-blocks** of 32 weights.

```
Super-block (256 weights):
├── d     (fp16): super-scale
├── dmin  (fp16): super-min
├── scales[0..7] (6-bit each): per sub-block scales
├── mins[0..7]   (6-bit each): per sub-block mins
└── 256 × 4-bit quantized values
```

For each sub-block i:
```
scale_i = d * sub_scale[i]
min_i   = dmin * sub_min[i]
weight  = scale_i * q - min_i
```

This gives each 32-weight sub-block its own adjusted scale AND min. The 6-bit sub-scales represent values 0..63. Multiply by the fp16 super-scale `d` to get the actual scale.

### Why K-Quants Are Better

| Format | Bits/Weight | Scales Stored At |
|--------|------------|-----------------|
| Q4_0 (naive) | 4.5 | Per 32 weights (1 scale) |
| Q4_K_M | 4.5 | Per 32 weights (sub-block), adjusted by super-block |

Same storage cost, but Q4_K_M's hierarchical structure means:
- Outlier weights in one sub-block don't hurt other sub-blocks
- The super-block scale handles the "typical magnitude" globally
- Sub-block scales handle local variation

Typical improvement in perplexity: Q4_K_M beats Q4_0 by 0.1-0.5 perplexity points.

---

## 9. The Memory/Quality Tradeoff Curve

Every quantization scheme is a point on this curve:

```
Perplexity
    │
 15 ┤ × Naive INT4 (Q4_0)
    │
 10 ┤
    │                    ← Quality cliff below 4-bit
  9 ┤ × GPTQ INT4
    │ × Q4_K_M
  8 ┤
    │ × Absmax INT8
  7 ┤ × FP16
    │ × FP32
    │
    └─────────────────────────────
    800  1100  2200  4400   (MB for TinyLlama-1.1B)
```

Key observations:
1. **FP32 vs FP16:** Same perplexity, 2× memory savings. Always use FP16.
2. **FP16 vs INT8:** Very small perplexity cost (~0.1-0.2), 2× memory savings. Almost always worth it.
3. **INT8 vs INT4 naive:** Large perplexity jump. NOT worth it without GPTQ/k-quants.
4. **INT8 vs INT4 GPTQ:** Small perplexity cost (~0.3-0.5), 2× memory savings. Worth it for deployment.
5. **Below 3-bit:** Quality falls off a cliff. Rarely used in practice.

The **Pareto frontier** is the curve connecting the best (lowest perplexity, lowest memory) points. GPTQ INT4 and Q4_K_M are both on or near this frontier.

---

## 10. Weight-Only vs Activation Quantization

So far we've only quantized **weights** (the model's parameters).

**Weights** are static — computed once, stored between requests.  
**Activations** are dynamic — computed fresh for every input token.

### Weight-Only Quantization (what we implement)

- Weights stored in INT4/INT8
- Dequantized to FP16 just before each matmul
- Activations stay in FP16 throughout
- Memory savings come from smaller weight storage

This is what GPTQ, k-quants, and our absmax/zeropoint implementations do.

**Tradeoff:** Memory savings without compute savings (the matmul still runs in FP16).

### Activation Quantization (not implemented here)

- Both weights AND activations stored in INT8
- The matmul itself runs in INT8 (using specialized hardware kernels)
- Saves memory AND speeds up computation
- Much harder because activations have **outlier features** in large LLMs

**The outlier problem:** LLMs develop activation values 100-1000× larger than typical values (called "emergent outlier features"). These outliers dominate the absmax scale, wasting most INT8 range on the typical values.

**LLM.int8()** (Dettmers et al.) solves this by decomposing the matmul:
```
Y = W_int8 @ X_int8_typical + W_fp16 @ X_fp16_outliers
```
The ~0.1% of outlier dimensions stay in FP16, everything else goes INT8.

This is more complex to implement — it's an extension idea for after you understand the basics in this project.
