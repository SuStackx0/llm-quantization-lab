# LLM Quantization Lab — Project Plan

> **Goal:** Build a from-scratch quantization toolkit that implements, applies, and benchmarks the core quantization techniques used in production LLM systems. This project completes an AI engineering trilogy:
> - `llm-serving-engine` → efficient inference scheduling (how models run fast)
> - `Investment-Intelligence-Platform` → agentic RAG applications (what LLMs build)
> - `llm-quantization-lab` → weight compression internals (how models are made small)

---

## Repository Structure

```
llm-quantization-lab/
├── README.md
├── requirements.txt
├── plan.md
├── src/
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── absmax.py              # Module 1: absmax quantizer
│   │   ├── zeropoint.py           # Module 1: zero-point quantizer
│   │   ├── gptq.py                # Module 2: GPTQ layer-wise quantization
│   │   ├── gptq_utils.py          # Module 2: Hessian computation helpers
│   │   └── dequantize.py          # Shared dequantization utilities
│   ├── gguf/
│   │   ├── __init__.py
│   │   ├── reader.py              # Module 3: GGUF file parser
│   │   ├── k_quants.py            # Module 3: k-quant block dequantization
│   │   └── loader.py              # Module 3: load GGUF tensors to torch
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── perplexity.py          # WikiText-2 perplexity evaluator
│   │   └── metrics.py             # Memory, latency, throughput helpers
│   └── api/
│       ├── __init__.py
│       └── server.py              # FastAPI server with /quantize endpoint
├── benchmarks/
│   ├── benchmark_quant.py         # Full precision vs quantized comparison
│   ├── benchmark_gguf.py          # GGUF format loading and inference speed
│   └── results/                   # Auto-generated JSON benchmark results
├── notebooks/
│   ├── 01_absmax_zeropoint.ipynb  # Module 1 interactive walkthrough
│   ├── 02_gptq_walkthrough.ipynb  # Module 2 interactive walkthrough
│   ├── 03_gguf_inspection.ipynb   # Module 3 GGUF format exploration
│   └── 04_benchmark_dashboard.ipynb  # Module 4 visualization dashboard
├── scripts/
│   ├── run_server.py              # Start the FastAPI server
│   ├── quantize_model.py          # CLI: quantize a HuggingFace model
│   └── eval_perplexity.py         # CLI: evaluate a model's perplexity
├── tests/
│   ├── test_absmax.py
│   ├── test_zeropoint.py
│   ├── test_gptq.py
│   ├── test_gguf_reader.py
│   └── test_perplexity.py
└── docs/
    ├── theory.md                  # Math behind each technique
    ├── gguf_format.md             # GGUF binary format walkthrough
    └── results_analysis.md        # Commentary on benchmark results
```

---

## Module 1 — Absmax & Zero-Point Quantization (INT8)

**Learning objective:** Understand the two foundational quantization schemes. These are the building blocks everything else is built on.

### What to implement

#### `src/quantization/absmax.py`

Symmetric quantization. Scale weights by the maximum absolute value so the range `[-max, max]` maps to `[-127, 127]`.

```python
# Interface to implement
def absmax_quantize(tensor: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, float]:
    """
    Returns (quantized_int_tensor, scale_factor).
    scale = max(|tensor|) / (2^(bits-1) - 1)
    quantized = round(tensor / scale).clamp(-127, 127).to(torch.int8)
    """

def absmax_dequantize(quantized: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Returns approximate original tensor: quantized.float() * scale
    """
```

**Why this matters:** Absmax is used in LLM.int8() for activations. It's simple but wastes range when weights are not zero-centered.

#### `src/quantization/zeropoint.py`

Asymmetric quantization. Adds a zero-point offset so the actual min/max of the tensor maps to `[0, 255]`. Better for non-symmetric weight distributions.

```python
def zeropoint_quantize(tensor: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, float, int]:
    """
    Returns (quantized_uint_tensor, scale, zero_point).
    scale = (max - min) / (2^bits - 1)
    zero_point = round(-min / scale)
    quantized = round(tensor / scale + zero_point).clamp(0, 255).to(torch.uint8)
    """

def zeropoint_dequantize(quantized: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """
    Returns approximate original: (quantized.float() - zero_point) * scale
    """
```

### Apply to a real model

In `notebooks/01_absmax_zeropoint.ipynb`:

1. Load TinyLlama-1.1B with `transformers` (you already have this from your serving engine)
2. Inspect weight distribution of `model.model.layers[0].self_attn.q_proj.weight`
3. Apply both quantizers to every `nn.Linear` weight in the model
4. Plot histograms: original FP32 weights vs quantized INT8 weights using matplotlib
5. Measure memory: `sys.getsizeof` / `tensor.element_size() * tensor.nelement()`
6. Run a sample generation — compare output quality before and after quantization

### Key concepts to document in `docs/theory.md`

- Quantization error = `dequantize(quantize(x)) - x` — measure and visualize this
- Why INT8 saves ~4× memory vs FP32 (4 bytes → 1 byte per weight)
- Symmetric vs asymmetric: when does zero-point help?
- Per-tensor vs per-channel quantization: apply absmax per-row on weight matrices and compare error

### Tests (`tests/test_absmax.py`, `tests/test_zeropoint.py`)

- Round-trip test: `dequantize(quantize(x))` should be close to `x` within tolerance
- Edge cases: all-zeros tensor, single-value tensor, negative-only tensor
- INT4 vs INT8: verify quantized values are within correct integer range
- Scale/zero_point output shapes and dtypes are correct

---

## Module 2 — GPTQ (Post-Training Quantization)

**Learning objective:** Understand why naive rounding loses quality at INT4, and how GPTQ uses second-order information (Hessians) to compensate. This is the algorithm behind AutoGPTQ, ExLlamaV2, and most INT4 GGUF models.

### Background (implement this understanding, not just copy)

GPTQ is from the paper "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2022). The core idea: when you quantize weight `w_i`, you're introducing error `δ_i`. Instead of ignoring it, you *compensate* by adjusting the remaining unquantized weights using the inverse Hessian of the layer's output loss.

The Hessian `H = 2XX^T` where `X` is the layer's input activations (collected during a calibration forward pass).

### What to implement

#### `src/quantization/gptq_utils.py`

```python
def collect_input_stats(model, calibration_data: list[torch.Tensor], layer_name: str) -> torch.Tensor:
    """
    Run calibration_data through the model with a forward hook on the target layer.
    Collect input activations X of shape [n_samples, seq_len, d_in].
    Return X reshaped to [d_in, n_tokens] for Hessian computation.
    """

def compute_hessian(X: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """
    H = 2 * X @ X.T / n_samples
    Add damping: H += damp * mean(diag(H)) * I
    Damping prevents numerical instability on near-singular Hessians.
    Returns H of shape [d_in, d_in].
    """

def cholesky_inverse(H: torch.Tensor) -> torch.Tensor:
    """
    Compute H_inv via Cholesky decomposition for numerical stability.
    torch.linalg.cholesky then torch.cholesky_inverse.
    """
```

#### `src/quantization/gptq.py`

```python
def gptq_quantize_layer(
    W: torch.Tensor,       # weight matrix [d_out, d_in]
    H_inv: torch.Tensor,   # inverse Hessian [d_in, d_in]
    bits: int = 4,
    block_size: int = 128, # process columns in blocks (memory efficiency)
    actorder: bool = False # sort columns by Hessian diagonal (improves quality)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (quantized_W, scales, zeros) per block.

    Algorithm (simplified OBC/GPTQ):
    For each column j in W:
        1. Quantize W[:, j] using per-column absmax → W_q[:, j]
        2. Compute error: e = W[:, j] - dequantize(W_q[:, j])
        3. Propagate error to remaining columns:
           W[:, j+1:] -= outer(e, H_inv[j, j+1:]) / H_inv[j, j]
    """
```

### Apply GPTQ to TinyLlama

In `scripts/quantize_model.py`:

1. Load TinyLlama-1.1B in FP16
2. Load 128 samples from WikiText-2 as calibration data
3. For each `nn.Linear` layer in the transformer:
   - Collect input activations via forward hook
   - Compute Hessian and its inverse
   - Run `gptq_quantize_layer` to get INT4 weights + scales + zeros
   - Replace the layer's weight with a `QuantizedLinear` module that stores INT4 + dequantizes on forward
4. Save the quantized model state dict

```python
class QuantizedLinear(nn.Module):
    """
    Stores weights as INT4 (packed into int8 tensors), scales, and zeros.
    Forward pass: dequantize on the fly to FP16 → matmul → output.
    This is "weight-only quantization" — activations stay FP16.
    """
    def __init__(self, in_features, out_features, bits=4):
        ...
    def pack(self, W_int: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
        # Pack two INT4 values into one INT8: saves 2× memory
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_fp16 = self.dequantize()
        return F.linear(x, W_fp16, self.bias)
```

### Perplexity evaluation (`src/eval/perplexity.py`)

```python
def compute_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    stride: int = 512,
    max_length: int = 1024,
) -> float:
    """
    Standard sliding-window perplexity on WikiText-2 test set.
    Lower = better. FP32 baseline ~5.5 for TinyLlama.
    Measure this for: FP32, FP16, absmax INT8, GPTQ INT4.
    """
```

### Key concepts to document

- Why naive INT4 rounding collapses quality (the quantization grid is too coarse)
- What the Hessian captures: how sensitive the output is to each weight
- Why the Cholesky trick matters: direct matrix inversion is numerically unstable
- Weight-only quantization vs activation quantization — GPTQ only quantizes weights

### Tests (`tests/test_gptq.py`)

- Single linear layer: GPTQ INT4 perplexity should be lower than naive INT4
- Hessian symmetry: `H == H.T` within float tolerance
- `QuantizedLinear` round-trip: pack then unpack should recover original INT4 values
- Output shapes unchanged after quantization

---

## Module 3 — GGUF Format & K-Quants

**Learning objective:** Understand the on-disk format used by llama.cpp and Ollama. Learn why k-quants (mixed precision per block) outperform uniform quantization.

### GGUF format overview (document in `docs/gguf_format.md`)

GGUF is a binary format with:
- A header: magic bytes `GGUF`, version, tensor count, metadata key-value pairs
- Metadata: model architecture, tokenizer vocab, hyperparameters
- Tensor data: name, shape, dtype, byte offset

K-quant types you will parse:
- `Q4_K_M` — 4-bit with mixed 6-bit for attention/embedding layers
- `Q5_K_S` — 5-bit small variant
- `Q8_0` — 8-bit with simple block scaling (baseline)

Each block in `Q4_K_M` is 256 weights with two scales (super-block + sub-block) and two min values. This is the key insight: by storing scales at block granularity rather than per-tensor, quantization error is much lower.

### What to implement

#### `src/gguf/reader.py`

```python
class GGUFReader:
    """
    Parse a GGUF file from disk without loading all tensors into memory.
    
    Usage:
        reader = GGUFReader("llama-3.2-1b.gguf")
        print(reader.metadata)              # dict of all key-value pairs
        print(reader.tensor_infos)          # list of TensorInfo(name, shape, dtype, offset)
        tensor = reader.load_tensor("blk.0.attn_q.weight")
    """
    
    MAGIC = b"GGUF"
    SUPPORTED_VERSIONS = [2, 3]
    
    def __init__(self, path: str):
        # Open file, verify magic and version, parse header
        ...
    
    def _parse_metadata(self) -> dict:
        # Read n_kv key-value pairs
        # Each entry: key (string), value_type (uint32), value
        # Value types: uint8=0, int8=1, uint16=2, ... string=8, array=9
        ...
    
    def _parse_tensor_infos(self) -> list[TensorInfo]:
        # Read n_tensors entries
        # Each: name (string), n_dims (uint32), dims ([uint64]*n_dims), dtype (uint32), offset (uint64)
        ...
    
    def load_tensor(self, name: str) -> torch.Tensor:
        # Seek to tensor's byte offset, read raw bytes, dispatch to k_quants.py for dequantization
        ...
```

#### `src/gguf/k_quants.py`

```python
def dequantize_q4_k(data: bytes, shape: tuple) -> torch.Tensor:
    """
    Q4_K block layout (256 weights per block):
    - 2 bytes: d (super-block scale, fp16)
    - 2 bytes: dmin (super-block min, fp16)
    - 12 bytes: scales and mins for 8 sub-blocks (6-bit packed)
    - 128 bytes: 256 × 4-bit quantized values (2 per byte)
    
    Dequantize:
    For each sub-block i (32 weights):
        scale = d * sub_scale[i]
        min   = dmin * sub_min[i]
        w[j]  = scale * q[j] - min
    Returns FP32 tensor of shape `shape`.
    """

def dequantize_q8_0(data: bytes, shape: tuple) -> torch.Tensor:
    """
    Q8_0 block layout (32 weights per block):
    - 2 bytes: d (block scale, fp16)
    - 32 bytes: 32 × int8 quantized values
    
    Dequantize: w[j] = d * q[j]
    Simpler than Q4_K — good starting point to implement first.
    """
```

#### `src/gguf/loader.py`

```python
class GGUFModelLoader:
    """
    Load a GGUF file and return a dict of dequantized tensors suitable for
    injecting into a HuggingFace model (maps GGUF tensor names → HF param names).
    """
    
    GGUF_TO_HF_NAME_MAP = {
        "blk.{i}.attn_q.weight": "model.layers.{i}.self_attn.q_proj.weight",
        "blk.{i}.attn_k.weight": "model.layers.{i}.self_attn.k_proj.weight",
        # ... full mapping
    }
    
    def load(self, gguf_path: str, model_config) -> dict[str, torch.Tensor]:
        ...
```

### Explore in `notebooks/03_gguf_inspection.ipynb`

1. Download `Llama-3.2-1B-Q4_K_M.gguf` (from HuggingFace Hub)
2. Use `GGUFReader` to print all metadata key-value pairs
3. Print all tensor names, shapes, and GGUF dtype strings
4. Load `blk.0.attn_q.weight` — show raw bytes, then dequantized FP32
5. Compare memory: raw GGUF file size vs FP32 equivalent (`params × 4 bytes`)
6. Visualize Q4_K block structure: plot scales across blocks for one layer

### Tests (`tests/test_gguf_reader.py`)

- Parse magic bytes and version correctly
- Metadata round-trip: string/integer/float values parsed correctly
- `dequantize_q8_0` against reference implementation (use llama.cpp Python bindings as oracle)
- Tensor shape inference from GGUF dims matches expected model shape

---

## Module 4 — Benchmark Dashboard

**Learning objective:** Measure the real tradeoffs so you can make informed decisions about which quantization scheme to use for a given deployment constraint.

### Metrics to measure (`src/eval/metrics.py`)

```python
@dataclass
class QuantizationResult:
    scheme: str              # "FP32", "FP16", "absmax_INT8", "zeropoint_INT8", "GPTQ_INT4", "GGUF_Q4_K_M"
    model_size_mb: float     # disk / in-memory footprint
    perplexity: float        # WikiText-2 test perplexity (lower = better)
    tokens_per_second: float # generation throughput
    ttft_ms: float           # time to first token (ms)
    peak_memory_mb: float    # peak RAM/MPS memory during generation
    quantization_time_s: float  # how long it took to quantize (0 for GGUF load)
```

### Benchmark scripts

#### `benchmarks/benchmark_quant.py`

```python
"""
Runs all quantization schemes on TinyLlama-1.1B and records QuantizationResult for each.
Saves to benchmarks/results/quant_results.json.

Schemes tested:
  1. FP32 baseline (HuggingFace default)
  2. FP16 (model.half())
  3. Absmax INT8 (our Module 1 impl)
  4. Zero-point INT8 (our Module 1 impl)
  5. GPTQ INT4 (our Module 2 impl)
  6. GPTQ INT4 with activation order (--actorder flag)

For each scheme:
  - Load model
  - Apply quantization
  - Measure peak memory (torch.mps.driver_allocated_memory() on Apple Silicon)
  - Run perplexity eval on 100 WikiText-2 samples (fast estimate)
  - Run 10 generations of 128 tokens, record TTFT and tokens/s
"""
```

#### `benchmarks/benchmark_gguf.py`

```python
"""
Load GGUF variants and compare against our implementations.

Tests:
  - Q8_0 (closest to our absmax INT8)
  - Q4_K_M (closest to our GPTQ INT4)
  - Q5_K_S

For each: measure load time, inference speed, memory footprint.
Cross-reference perplexity with our own implementations.
"""
```

### Dashboard notebook (`notebooks/04_benchmark_dashboard.ipynb`)

Produce 4 charts using matplotlib/plotly:

**Chart 1: Memory vs Perplexity scatter plot**
- X-axis: model size in MB
- Y-axis: perplexity
- Each scheme is a labeled point
- Pareto frontier highlighted — you want low memory AND low perplexity

**Chart 2: Throughput bar chart**
- X-axis: quantization scheme
- Y-axis: tokens/second
- Color-code by memory footprint

**Chart 3: Quantization error heatmap**
- For a single attention layer in TinyLlama
- X/Y-axes: weight matrix rows/columns
- Color: `|W_original - W_dequantized|`
- Show this for: absmax INT8, GPTQ INT4, GGUF Q4_K
- Side-by-side comparison showing GPTQ has lower error than naive INT4

**Chart 4: Perplexity vs bits-per-weight curve**
- X-axis: effective bits per weight (32, 16, 8, 4)
- Y-axis: perplexity
- Shows the quality cliff below 4-bit

### FastAPI integration (`src/api/server.py`)

Extends your existing serving engine's API pattern with a `/quantize` endpoint:

```python
POST /quantize
Body: {
  "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "scheme": "gptq_int4" | "absmax_int8" | "zeropoint_int8",
  "calibration_samples": 128,
  "save_path": "./quantized_models/"
}
Response: {
  "status": "done",
  "original_size_mb": 4400,
  "quantized_size_mb": 850,
  "compression_ratio": 5.2,
  "perplexity_delta": 0.3
}

GET /quantize/schemes
Response: list of available quantization schemes with expected compression ratios

GET /quantize/results
Response: latest benchmark results from benchmarks/results/quant_results.json
```

This endpoint is designed to be called by your `llm-serving-engine` — it quantizes a model on demand so the serving engine can load a smaller version.

---

## Integration with llm-serving-engine

The explicit connection between your two projects:

In `llm-serving-engine`, add an optional flag to `run_server.py`:

```bash
python scripts/run_server.py --device mps --quantize absmax_int8
```

Under the hood, before starting the server, this calls the `/quantize` endpoint of `llm-quantization-lab` (if running) or applies the quantization inline. The serving engine then loads the quantized weights instead of FP32.

Benchmark the combined effect:
- Original: FP32 model + PagedAttention serving engine
- Quantized: INT8 model + PagedAttention serving engine
- Measure: TTFT, throughput, memory — the quantization benefit is additive with batching

Document this in `docs/results_analysis.md` as the capstone finding.

---

## Implementation Order

### Week 1 — Module 1 (Absmax & Zero-Point)

| Day | Task |
|-----|------|
| 1 | Set up repo structure, `requirements.txt`, implement `absmax_quantize` and `absmax_dequantize` |
| 2 | Implement `zeropoint_quantize` and `zeropoint_dequantize` |
| 3 | Apply both to TinyLlama, measure memory, write `tests/test_absmax.py` and `tests/test_zeropoint.py` |
| 4 | Build `notebooks/01_absmax_zeropoint.ipynb` with weight distribution visualizations |
| 5 | Implement `src/eval/perplexity.py`, run first perplexity numbers, write `docs/theory.md` section 1 |

### Week 2 — Module 2 (GPTQ)

| Day | Task |
|-----|------|
| 6 | Implement `collect_input_stats` and forward hooks |
| 7 | Implement `compute_hessian` and `cholesky_inverse` |
| 8 | Implement `gptq_quantize_layer` — the core algorithm |
| 9 | Implement `QuantizedLinear` with INT4 packing/unpacking |
| 10 | Apply GPTQ to full TinyLlama, measure perplexity, write tests |

### Week 3 — Modules 3 & 4 (GGUF + Benchmarks)

| Day | Task |
|-----|------|
| 11 | Implement `GGUFReader` — parse header, metadata, tensor infos |
| 12 | Implement `dequantize_q8_0` and `dequantize_q4_k` |
| 13 | Implement `GGUFModelLoader`, write `notebooks/03_gguf_inspection.ipynb` |
| 14 | Run full benchmark suite, write `benchmarks/benchmark_quant.py` |
| 15 | Build dashboard notebook, implement FastAPI server, write final README |

---

## Requirements

```
# Core
torch>=2.2.0
transformers>=4.40.0
datasets>=2.18.0          # WikiText-2 for perplexity
accelerate>=0.29.0        # Model loading utilities
safetensors>=0.4.3

# GGUF
gguf>=0.6.0               # Reference GGUF parser (for cross-checking)
huggingface-hub>=0.22.0   # Download GGUF models

# API
fastapi>=0.110.0
uvicorn>=0.29.0

# Visualization & Notebooks
matplotlib>=3.8.0
plotly>=5.20.0
jupyter>=1.0.0
ipywidgets>=8.1.0

# Testing
pytest>=8.1.0
pytest-cov>=5.0.0
```

---

## README Narrative

The README should tell this story:

> "I built a from-scratch quantization toolkit to understand how production LLM systems compress models from FP32 down to INT4. Starting with the math of absmax and zero-point quantization, I then implemented a simplified GPTQ (the algorithm behind AutoGPTQ and most INT4 GGUF models) and built a GGUF binary format parser to understand how llama.cpp stores k-quants on disk.
>
> The result: TinyLlama-1.1B goes from 4.4GB (FP32) to 850MB (GPTQ INT4) with only +0.3 perplexity degradation, while running 2.1× faster on MPS. Plugged into my llm-serving-engine, the quantized model handles the same concurrent request load at 40% of the memory footprint."

Headline benchmark table:

| Scheme | Size (MB) | Perplexity | Tokens/s | Memory (MB) |
|--------|-----------|------------|----------|-------------|
| FP32 | 4,400 | ~5.5 | baseline | ~4,600 |
| FP16 | 2,200 | ~5.5 | +15% | ~2,300 |
| Absmax INT8 | 1,100 | ~5.7 | +30% | ~1,200 |
| GPTQ INT4 | 850 | ~5.8 | +60% | ~950 |
| GGUF Q4_K_M | 800 | ~5.75 | +65% | ~900 |

*(Fill in real numbers when you run it — these are approximate targets.)*

---

## Concepts You Will Deeply Understand After This Project

1. **Why INT4 is the sweet spot** — below 4-bit, perplexity degrades sharply; above 8-bit, memory savings don't justify the overhead
2. **Why GPTQ beats naive rounding** — the Hessian tells you which weights matter most for output quality; you can afford to be imprecise on low-importance weights
3. **Why k-quants (Q4_K_M) beat uniform INT4** — mixed precision within a super-block adapts to local weight variance
4. **Weight-only vs activation quantization** — GPTQ only quantizes weights; LLM.int8() also handles activations with outlier decomposition
5. **The memory/quality Pareto frontier** — every deployment decision is a point on this curve; now you can measure it yourself
6. **How llama.cpp/Ollama work internally** — GGUF is the format they use; you can now read it from scratch

---

## Extension Ideas (post-MVP)

- **AWQ (Activation-aware Weight Quantization):** Similar to GPTQ but uses activation magnitude to weight the importance of each channel — often better quality at INT4
- **GGML tensor type browser:** Build a CLI tool that pretty-prints any GGUF file's tensor inventory with sizes and types
- **Quantization-aware fine-tuning stub:** Add fake quantization nodes during training so the model learns to be robust to quantization error
- **Serve both engines together:** Have `llm-serving-engine` call `llm-quantization-lab`'s `/quantize` endpoint at startup if a quantized checkpoint doesn't already exist
