# LLM Quantization Lab

A from-scratch implementation of LLM quantization techniques — built to understand how production systems compress models from FP32 down to INT4.

> TinyLlama-1.1B goes from 4.4 GB (FP32) to ~850 MB (GPTQ INT4) with only a small perplexity increase, while running faster on memory-bandwidth-limited hardware.

---

## What You'll Learn

By reading the docs, running the notebooks, and reading the code, you'll understand:

1. **Why INT4 is the sweet spot** — quality falls off a cliff below 4-bit
2. **Why GPTQ beats naive rounding** — the Hessian tells you which weights matter most
3. **Why k-quants (Q4_K_M) are so popular** — mixed precision per block adapts to local weight variance
4. **How llama.cpp / Ollama work internally** — GGUF is the format they use; you can now read it from scratch
5. **The real memory/quality tradeoff** — measured on a real model, not just theoretical

---

## Learning Path

**Start here → follow in order:**

```
1. docs/theory.md         ← Math behind every technique (read this first)
2. src/quantization/absmax.py      ← Simplest quantization scheme
3. notebooks/01_absmax_zeropoint.ipynb  ← Apply to TinyLlama, see it work
4. docs/theory.md section 7       ← GPTQ math
5. src/quantization/gptq.py       ← GPTQ implementation
6. notebooks/02_gptq_walkthrough.ipynb  ← Compare naive vs GPTQ
7. docs/gguf_format.md            ← GGUF binary format explanation
8. src/gguf/reader.py             ← Parse a real .gguf file
9. notebooks/03_gguf_inspection.ipynb   ← Explore a real GGUF model
10. benchmarks/benchmark_quant.py ← Run the full comparison
11. notebooks/04_benchmark_dashboard.ipynb ← Visualize the tradeoffs
12. docs/results_analysis.md      ← Interpret what you measured
```

---

## Project Structure

```
LLM-Quantization/
├── src/
│   ├── quantization/
│   │   ├── absmax.py         ← Symmetric INT8/INT4 quantization
│   │   ├── zeropoint.py      ← Asymmetric INT8 quantization
│   │   ├── gptq_utils.py     ← Hessian computation helpers
│   │   ├── gptq.py           ← GPTQ algorithm + QuantizedLinear
│   │   └── dequantize.py     ← Shared dequantization
│   ├── gguf/
│   │   ├── reader.py         ← GGUF file parser
│   │   ├── k_quants.py       ← Q8_0, Q4_K, Q5_K dequantization
│   │   └── loader.py         ← GGUF → HuggingFace name mapping
│   ├── eval/
│   │   ├── perplexity.py     ← WikiText-2 perplexity evaluator
│   │   └── metrics.py        ← Memory, speed, TTFT measurement
│   └── api/
│       └── server.py         ← FastAPI /quantize endpoint
├── notebooks/
│   ├── 01_absmax_zeropoint.ipynb     ← Module 1 walkthrough
│   ├── 02_gptq_walkthrough.ipynb     ← Module 2 walkthrough
│   ├── 03_gguf_inspection.ipynb      ← Module 3 walkthrough
│   └── 04_benchmark_dashboard.ipynb  ← Charts and analysis
├── benchmarks/
│   ├── benchmark_quant.py    ← Compare all schemes on TinyLlama
│   └── benchmark_gguf.py     ← GGUF loading speed test
├── scripts/
│   ├── quantize_model.py     ← CLI: quantize any HF model
│   ├── eval_perplexity.py    ← CLI: measure perplexity
│   └── run_server.py         ← Start the FastAPI server
├── tests/
│   ├── test_absmax.py
│   ├── test_zeropoint.py
│   ├── test_gptq.py
│   └── test_gguf_reader.py
└── docs/
    ├── theory.md             ← Complete quantization theory (start here)
    ├── gguf_format.md        ← GGUF binary format deep-dive
    └── results_analysis.md   ← How to interpret benchmark numbers
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (no model download needed)
pytest tests/ -v

# Quantize TinyLlama with absmax INT8
python scripts/quantize_model.py --scheme absmax_int8

# Evaluate perplexity
python scripts/eval_perplexity.py --scheme fp16 --quick
python scripts/eval_perplexity.py --scheme absmax_int8 --quick

# Run benchmarks (takes 10-30 min)
python benchmarks/benchmark_quant.py --quick

# Start the API server
python scripts/run_server.py
# Then visit http://localhost:8000/docs
```

---

## Benchmark Results

*(Fill in your actual numbers after running the benchmarks)*

| Scheme | Size (MB) | Perplexity | Tokens/s | Peak Mem (MB) |
|--------|-----------|------------|----------|---------------|
| FP32 | 4,400 | ~7.8 | baseline | ~4,600 |
| FP16 | 2,200 | ~7.8 | +15% | ~2,300 |
| Absmax INT8 | ~1,100* | ~8.0 | +20% | ~1,200 |
| GPTQ INT4 | ~850 | ~8.3 | +35% | ~950 |

*\*When using QuantizedLinear with actual INT4 packing*

---

## Module Overview

### Module 1: Absmax & Zero-Point Quantization
**Files:** `src/quantization/absmax.py`, `src/quantization/zeropoint.py`

The two fundamental quantization schemes. Absmax is symmetric (maps `[-max, max]` to INT8). Zero-point is asymmetric (maps `[min, max]` to UINT8 — better for skewed distributions). These are the building blocks for everything else.

### Module 2: GPTQ (Post-Training Quantization)
**Files:** `src/quantization/gptq.py`, `src/quantization/gptq_utils.py`

The algorithm behind AutoGPTQ and most INT4 GGUF models. After quantizing each column of a weight matrix, it propagates the rounding error to compensate remaining columns — guided by the inverse Hessian computed from calibration activations. Result: INT4 quality much closer to INT8 than naive rounding achieves.

### Module 3: GGUF Format & K-Quants
**Files:** `src/gguf/reader.py`, `src/gguf/k_quants.py`, `src/gguf/loader.py`

A from-scratch parser for the GGUF binary format used by llama.cpp and Ollama. Includes dequantization for Q8_0, Q4_K (M/S), and Q5_K. After this module, you can open any `.gguf` file in Python and read individual tensors.

### Module 4: Benchmarks & Dashboard
**Files:** `benchmarks/benchmark_quant.py`, `notebooks/04_benchmark_dashboard.ipynb`

Run all schemes on TinyLlama and produce four charts: Memory vs Perplexity scatter, speed comparison, error heatmaps, and the perplexity-vs-bits curve showing the 4-bit quality cliff.

---

## Key Concepts

**Why quantization works at all:**
Neural networks are surprisingly robust to weight precision. Weights encode *directions* in high-dimensional space more than exact magnitudes. Rounding to INT8 changes directions only slightly, and the network's redundancy absorbs the error.

**Why INT4 needs GPTQ:**
With only 16 distinct values per weight, rounding errors are ~18× larger than INT8. These errors compound across layers. GPTQ uses second-order information (the Hessian) to redistribute errors to weights that can absorb them without affecting output quality.

**Why k-quants beat uniform INT4:**
Each 256-weight super-block in Q4_K_M gets its own scale and min, adapted to that block's actual value range. This prevents outlier weights in one block from degrading the precision of all other blocks — the same problem that GPTQ solves, but using a simpler block-adaptive approach.

---

## API Usage

```bash
# Start server
python scripts/run_server.py

# Quantize a model
curl -X POST http://localhost:8000/quantize \
  -H "Content-Type: application/json" \
  -d '{"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "scheme": "absmax_int8"}'

# List available schemes
curl http://localhost:8000/quantize/schemes

# Get benchmark results
curl http://localhost:8000/quantize/results
```
