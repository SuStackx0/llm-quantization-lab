"""
Benchmark: Compare All Quantization Schemes
=============================================

Runs all quantization schemes on TinyLlama-1.1B and records metrics for each.
Saves results to benchmarks/results/quant_results.json.

Usage:
    python benchmarks/benchmark_quant.py
    python benchmarks/benchmark_quant.py --quick   # faster, less accurate
    python benchmarks/benchmark_quant.py --schemes fp16 absmax_int8   # specific schemes

This will take 10-30 minutes to run completely (GPTQ is the slow step).
Progress is printed as each scheme completes.
"""

import sys
import os
import json
import time
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.perplexity import compute_perplexity, quick_perplexity
from src.eval.metrics import QuantizationResult, measure_model_size_mb, measure_generation_speed, measure_peak_memory_mb
from src.quantization.absmax import absmax_quantize, absmax_dequantize
from src.quantization.zeropoint import zeropoint_quantize, zeropoint_dequantize

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
RESULTS_PATH = "benchmarks/results/quant_results.json"


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    elif torch.cuda.is_available(): return "cuda"
    return "cpu"


def load_fresh_model(dtype=torch.float16, device="cpu"):
    """Load a fresh copy of TinyLlama from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map=device
    )
    return model, tokenizer


def run_benchmark_for_scheme(scheme: str, quick: bool, device: str) -> QuantizationResult:
    """Load model, apply quantization, measure all metrics."""
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {scheme}")
    print(f"{'='*60}")

    result = QuantizationResult(scheme=scheme)

    # Load model
    if scheme == "fp32":
        dtype = torch.float32
    else:
        dtype = torch.float16

    print(f"  Loading model ({dtype})...")
    model, tokenizer = load_fresh_model(dtype=dtype, device=device)
    result.model_size_mb = measure_model_size_mb(model)
    print(f"  Original size: {result.model_size_mb:.1f} MB")

    # Apply quantization
    t_quant_start = time.time()

    if scheme in ("fp32", "fp16"):
        pass  # No quantization needed

    elif scheme == "absmax_int8":
        print("  Applying absmax INT8...")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float()
                q, scale = absmax_quantize(W, bits=8)
                module.weight.data = absmax_dequantize(q, scale).to(dtype)
        result.model_size_mb = measure_model_size_mb(model)

    elif scheme == "zeropoint_int8":
        print("  Applying zero-point INT8...")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float()
                q, scale, zp = zeropoint_quantize(W, bits=8)
                module.weight.data = zeropoint_dequantize(q, scale, zp).to(dtype)
        result.model_size_mb = measure_model_size_mb(model)

    elif scheme == "gptq_int4":
        print("  Applying GPTQ INT4 (this takes a few minutes)...")
        from src.quantization.gptq import gptq_quantize_layer, QuantizedLinear
        from src.quantization.gptq_utils import collect_input_stats, compute_hessian, cholesky_inverse
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        texts = [t for t in dataset["text"] if len(t.strip()) > 50][:128]
        calib_tokens = tokenizer(
            texts, return_tensors="pt", max_length=512, truncation=True, padding=True
        ).input_ids.to(device)

        replacements = {}
        model.eval()
        linear_layers = [
            (name, mod) for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and "lm_head" not in name
        ]

        for i, (name, module) in enumerate(linear_layers):
            print(f"    [{i+1}/{len(linear_layers)}] {name}", flush=True)
            try:
                X = collect_input_stats(model, calib_tokens, name, device=device)
                H = compute_hessian(X, damp=0.01)
                H_inv = cholesky_inverse(H)
                W = module.weight.data.float()
                W_q, scales, zeros = gptq_quantize_layer(W, H_inv, bits=4)
                q_linear = QuantizedLinear.from_float(module, W_q.to(torch.int8), scales, zeros, bits=4)
                replacements[name] = q_linear
            except Exception as e:
                print(f"      Failed: {e}")

        for name, new_mod in replacements.items():
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_mod)

        result.model_size_mb = measure_model_size_mb(model)

    result.quantization_time_s = round(time.time() - t_quant_start, 1)
    print(f"  Quantized size: {result.model_size_mb:.1f} MB")
    print(f"  Quantization time: {result.quantization_time_s}s")

    # Measure perplexity
    print("  Computing perplexity...")
    try:
        if quick:
            result.perplexity = quick_perplexity(model, tokenizer, device=device)
        else:
            result.perplexity = compute_perplexity(model, tokenizer, device=device)
        print(f"  Perplexity: {result.perplexity:.2f}")
    except Exception as e:
        print(f"  Perplexity failed: {e}")

    # Measure generation speed
    print("  Measuring generation speed...")
    try:
        result.tokens_per_second, result.ttft_ms = measure_generation_speed(
            model, tokenizer, n_tokens=32, n_runs=2, device=device
        )
        print(f"  Speed: {result.tokens_per_second:.1f} tok/s, TTFT: {result.ttft_ms:.1f} ms")
    except Exception as e:
        print(f"  Speed measurement failed: {e}")

    # Peak memory
    result.peak_memory_mb = measure_peak_memory_mb(device)
    print(f"  Peak memory: {result.peak_memory_mb:.1f} MB")

    return result


def print_summary(results: list[QuantizationResult]):
    """Print a comparison table of all results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    header = f"{'Scheme':<20} {'Size(MB)':>10} {'Perplexity':>12} {'Tok/s':>8} {'TTFT(ms)':>10} {'Mem(MB)':>10}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r.scheme:<20} {r.model_size_mb:>10.1f} {r.perplexity:>12.2f} "
              f"{r.tokens_per_second:>8.1f} {r.ttft_ms:>10.1f} {r.peak_memory_mb:>10.1f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast perplexity estimate")
    parser.add_argument("--schemes", nargs="+",
                        default=["fp16", "absmax_int8", "zeropoint_int8"],
                        choices=["fp32", "fp16", "absmax_int8", "zeropoint_int8", "gptq_int4"],
                        help="Which schemes to benchmark (gptq_int4 is slow)")
    args = parser.parse_args()

    device = get_device()
    print(f"Running benchmarks on device: {device}")
    print(f"Schemes: {args.schemes}")

    results = []
    for scheme in args.schemes:
        result = run_benchmark_for_scheme(scheme, args.quick, device)
        results.append(result)

        # Save intermediate results after each scheme
        os.makedirs("benchmarks/results", exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump([vars(r) for r in results], f, indent=2)
        print(f"  Saved intermediate results to {RESULTS_PATH}")

    print_summary(results)
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
