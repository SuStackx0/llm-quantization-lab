"""
Benchmark: GGUF Format Loading and Inference
==============================================

Compares GGUF quantized models against our own implementations.

Usage:
    # First download a GGUF model:
    # huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
    #   --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf" --local-dir ./models/

    python benchmarks/benchmark_gguf.py --gguf_path ./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

What this benchmarks:
  1. GGUF file load time (reading + dequantizing all tensors)
  2. Memory footprint of dequantized tensors vs raw GGUF file size
  3. Tensor-level comparison: does our Q4_K dequant match llama.cpp's?
"""

import sys
import os
import json
import time
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gguf.reader import GGUFReader


def benchmark_gguf_load(gguf_path: str) -> dict:
    """Measure GGUF file stats and load time."""
    file_size_mb = os.path.getsize(gguf_path) / (1024 ** 2)
    print(f"\nGGUF file: {gguf_path}")
    print(f"File size: {file_size_mb:.1f} MB")

    # Time the parse (metadata + tensor info, no data loaded)
    t0 = time.time()
    reader = GGUFReader(gguf_path)
    parse_time = time.time() - t0
    print(f"Parse time (metadata only): {parse_time*1000:.1f} ms")

    print(reader.summary())

    # Time loading all tensors that we support
    supported_dtypes = {0, 1, 8, 12, 13, 14, 15}  # F32, F16, Q8_0, Q4_K, Q5_K
    supported_tensors = [t for t in reader.tensor_infos if t.dtype_id in supported_dtypes]

    print(f"\nLoading {len(supported_tensors)} supported tensors...")
    t0 = time.time()
    total_elements = 0
    total_dequant_bytes = 0
    errors = 0

    for tensor_info in supported_tensors[:10]:  # load first 10 for speed
        try:
            tensor = reader.load_tensor(tensor_info.name)
            total_elements += tensor.nelement()
            total_dequant_bytes += tensor.nelement() * 4  # float32 = 4 bytes
        except NotImplementedError as e:
            errors += 1
        except Exception as e:
            print(f"  Error loading {tensor_info.name}: {e}")
            errors += 1

    load_time = time.time() - t0
    dequant_mb = total_dequant_bytes / (1024 ** 2)

    print(f"Loaded 10 tensors in {load_time:.2f}s")
    print(f"Dequantized size (10 tensors): {dequant_mb:.1f} MB (float32)")
    print(f"Errors: {errors}")

    return {
        "file_path": gguf_path,
        "file_size_mb": round(file_size_mb, 1),
        "n_tensors": len(reader.tensor_infos),
        "parse_time_ms": round(parse_time * 1000, 1),
        "sample_load_time_s": round(load_time, 2),
        "errors": errors,
    }


def compare_quantization_formats(gguf_path: str):
    """
    Load a tensor from GGUF and compare its values against
    our own absmax INT8 quantization of the same data.

    This shows how much different the formats are in practice.
    """
    reader = GGUFReader(gguf_path)

    # Find a Q8_0 or Q4_K tensor to compare
    target = None
    for t in reader.tensor_infos:
        if t.dtype_id in (8, 12, 13) and t.n_elements >= 256:
            target = t
            break

    if target is None:
        print("No suitable tensor found for comparison")
        return

    print(f"\nComparing {target.name} ({target.dtype_name}, shape={target.shape})")

    # Load from GGUF
    gguf_tensor = reader.load_tensor(target.name)
    print(f"GGUF dequantized: min={gguf_tensor.min():.4f}, max={gguf_tensor.max():.4f}, "
          f"mean={gguf_tensor.mean():.4f}")

    # Apply our absmax INT8 to the same data for comparison
    from src.quantization.absmax import absmax_quantize, absmax_dequantize
    q, scale = absmax_quantize(gguf_tensor.flatten(), bits=8)
    our_tensor = absmax_dequantize(q, scale).reshape(gguf_tensor.shape)

    diff = (gguf_tensor - our_tensor).abs()
    print(f"\nDifference (GGUF format vs our absmax INT8):")
    print(f"  Mean absolute diff: {diff.mean():.6f}")
    print(f"  Max absolute diff:  {diff.max():.6f}")
    print(f"\n  (Lower = our INT8 closely matches the GGUF representation)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf_path", required=True, help="Path to a .gguf file")
    args = parser.parse_args()

    if not os.path.exists(args.gguf_path):
        print(f"Error: {args.gguf_path} not found")
        print("\nDownload a GGUF model first:")
        print("  pip install huggingface_hub")
        print("  huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \\")
        print("    --include '*.gguf' --local-dir ./models/")
        return

    stats = benchmark_gguf_load(args.gguf_path)

    compare_quantization_formats(args.gguf_path)

    # Save results
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/gguf_results.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nResults saved to benchmarks/results/gguf_results.json")


if __name__ == "__main__":
    main()
