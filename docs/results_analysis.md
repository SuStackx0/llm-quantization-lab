# Results Analysis: What the Benchmarks Tell Us

> Fill this in after running `python benchmarks/benchmark_quant.py`.
> The analysis below explains what to expect and how to interpret the numbers.

---

## How to Run the Benchmarks

```bash
# Install dependencies
pip install -r requirements.txt

# Quick benchmark (faster, less accurate perplexity)
python benchmarks/benchmark_quant.py --quick --schemes fp16 absmax_int8 zeropoint_int8

# Full benchmark (takes 20-40 minutes, includes GPTQ)
python benchmarks/benchmark_quant.py --schemes fp16 absmax_int8 zeropoint_int8 gptq_int4

# View results
cat benchmarks/results/quant_results.json
```

---

## Expected Results (TinyLlama-1.1B)

After running benchmarks, fill in your actual numbers in this table:

| Scheme | Size (MB) | Perplexity | Tokens/s | TTFT (ms) | Peak Mem (MB) |
|--------|-----------|------------|----------|-----------|---------------|
| FP16 | ~2,200 | ~7-8 | baseline | baseline | ~2,300 |
| Absmax INT8 | ~2,200* | ~7-9 | similar | similar | ~2,300 |
| ZeroPoint INT8 | ~2,200* | ~7-9 | similar | similar | ~2,300 |
| GPTQ INT4 | ~1,100** | ~8-11 | +10-30% | -10-20% | ~1,200 |

*Note: Absmax/zeropoint INT8 in this implementation dequantize weights before storage
(so size stays the same). A production INT8 implementation would store int8 tensors
and dequantize on-the-fly, giving the actual 2× memory saving.

**GPTQ INT4 uses QuantizedLinear which packs INT4 into INT8 tensors, giving real size savings.

### Your Actual Numbers:
*(Run the benchmarks and fill in here)*

---

## How to Interpret Each Metric

### Model Size (MB)
This is `sum(param.nelement() * param.element_size() for param in model.parameters())`.

- FP16: 2 bytes × 1.1B params = **2.2 GB**
- INT8 (real): 1 byte × 1.1B params = **1.1 GB**
- INT4 (real): 0.5 bytes × 1.1B params = **0.55 GB**

For our QuantizedLinear, two INT4 values are packed per byte, so the weight storage is half of INT8.

### Perplexity
Lower = better. The formula is:

```
perplexity = exp(average cross-entropy loss on WikiText-2 test set)
```

**Reference values for TinyLlama-1.1B:**
- FP32/FP16: ~7.5-8.5 (this is the baseline quality)
- INT8: typically stays within 0.1-0.3 of baseline
- GPTQ INT4: typically within 0.3-1.0 of baseline (great for the 2× memory saving!)
- Naive INT4: can be 5-20 points higher (catastrophically bad)

**The perplexity cliff:** Below ~4 bits, perplexity degrades sharply. This is why INT4 is the practical floor for production deployments — 3-bit and 2-bit quantization is experimental.

### Tokens Per Second
Generation throughput. Measured as new tokens generated per second.

Why quantized models can be faster:
1. **Memory bandwidth:** Smaller weights → fewer bytes transferred from RAM to GPU per step
2. **Cache efficiency:** More weights fit in L2/L3 cache
3. **Dequantization overhead:** Costs a few operations per weight (small compared to matmul)

On CPU/MPS, bandwidth is often the bottleneck, so INT4 gives real speedups.
On high-end CUDA GPUs, compute is the bottleneck and the speedup depends on whether INT4 kernels are available.

### Time to First Token (TTFT)
How long until the first token appears. Depends mainly on the prefill computation speed.

Quantized models: faster prefill because weights load faster from memory.

### Peak Memory
Maximum RAM allocated during generation. This is what determines whether you can run the model at all on your hardware.

---

## What to Look For in Your Charts

After running the benchmark, generate the dashboard:

```bash
jupyter notebook notebooks/04_benchmark_dashboard.ipynb
```

### Chart 1: Memory vs Perplexity Scatter

Look for the **Pareto frontier** — the set of schemes where you can't improve one metric without worsening the other.

Expected pattern:
```
Perplexity
    ↑
11  |  × Naive INT4 (bad)
    |
 9  |      × GPTQ INT4 ←─ Pareto frontier
    |          × Q4_K_M
 8  |              × INT8
 7.5|                   × FP16 ─────────────── × FP32
    └──────────────────────────────────────────────────→ Size (MB)
        800         1100        2200      4400
```

If GPTQ INT4 falls on the Pareto frontier (low perplexity AND low memory), it wins.

### Chart 3: Quantization Error Heatmap

For a single attention layer, the error map shows WHERE in the weight matrix the most error occurs.

What to look for:
- **GPTQ INT4** should have lower maximum error than **Naive INT4** (absmax, 4-bit)
- The error pattern for absmax is uniform (same error everywhere)
- The error pattern for GPTQ should be lower in rows corresponding to high-Hessian-diagonal columns

### Chart 4: Perplexity vs Bits

```
Perplexity
    ↑
 50 |
    |
 20 | × 2-bit
    |
 12 |    × 3-bit
    |
  9 |         × 4-bit naive
    |              × 4-bit GPTQ
  8 |                  × 8-bit
7.5 |                       × 16-bit = 32-bit
    └──────────────────────────────────────────→ Bits per weight
       2      3      4       8      16
```

The key finding: there's a **sharp quality cliff** below 4 bits. This is why virtually all production quantized LLMs use 4-bit or higher.

---

## Why These Numbers Matter for Production Deployments

### Scenario: Deploying TinyLlama on a 16GB MacBook Pro

| Scheme | Can it run? | Notes |
|--------|-------------|-------|
| FP32 | Barely (needs 4.4 GB just for weights) | Slow, uses most RAM |
| FP16 | Yes (2.2 GB weights) | Good baseline |
| INT8 | Yes (1.1 GB weights) | More breathing room for context |
| INT4 GPTQ | Yes (0.55 GB weights) | Run 4x longer contexts! |

With INT4, you can use 4× longer context windows for the same memory budget, or run inference faster because weights fit in CPU cache.

### Scenario: Serving 100 concurrent users on one A100 (80GB)

| Scheme | Max model size | Notes |
|--------|---------------|-------|
| FP16 | 40B parameters | ~Half the A100 for model |
| INT8 | 80B parameters | Just fits Llama-70B! |
| INT4 | 160B parameters | Could fit future 100B+ models |

Quantization isn't just about making models run on small hardware — it's about fitting **bigger models** on the same hardware.

---

## Key Takeaways

After running and analyzing the benchmarks, you should understand:

1. **INT8 absmax is almost free** — tiny perplexity cost, 2× memory saving. Always worth it.

2. **GPTQ INT4 is the production sweet spot** — ~0.5 perplexity degradation, 4× smaller than FP16, ~20% faster on memory-bandwidth-limited hardware.

3. **The Hessian matters** — Naive INT4 has roughly 3× worse perplexity than GPTQ INT4 despite using the same number of bits. The difference is entirely due to error compensation.

4. **k-quants (Q4_K_M) match GPTQ** — llama.cpp's format achieves similar quality to GPTQ INT4 at the same bit width. The hierarchical scale structure does the same job as Hessian compensation, just with a different mechanism.

5. **Below 4-bit is experimental** — the quality cliff is real. 3-bit requires highly specialized techniques (like SpQR or QuIP#) and is not production-ready as of 2024.
