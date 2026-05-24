"""
CLI Script: Evaluate Perplexity
=================================

Usage:
    # FP16 baseline
    python scripts/eval_perplexity.py --scheme fp16

    # After applying quantization
    python scripts/eval_perplexity.py --scheme absmax_int8
    python scripts/eval_perplexity.py --scheme gptq_int4

    # Quick estimate (faster, less accurate)
    python scripts/eval_perplexity.py --scheme fp16 --quick
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.eval.perplexity import compute_perplexity, quick_perplexity
from src.quantization.absmax import absmax_quantize, absmax_dequantize
from src.quantization.zeropoint import zeropoint_quantize, zeropoint_dequantize
import torch.nn as nn


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--scheme", choices=["fp32", "fp16", "absmax_int8", "zeropoint_int8"],
                        default="fp16")
    parser.add_argument("--quick", action="store_true", help="Quick estimate using validation set")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load model in the right dtype
    dtype = torch.float16 if args.scheme != "fp32" else torch.float32
    print(f"\nLoading {args.model} in {dtype}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device,
    )

    # Apply quantization if needed
    if args.scheme == "absmax_int8":
        print("Applying absmax INT8 quantization...")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float()
                q, scale = absmax_quantize(W)
                module.weight.data = absmax_dequantize(q, scale).to(dtype)

    elif args.scheme == "zeropoint_int8":
        print("Applying zeropoint INT8 quantization...")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float()
                q, scale, zp = zeropoint_quantize(W)
                module.weight.data = zeropoint_dequantize(q, scale, zp).to(dtype)

    # Compute perplexity
    print(f"\nComputing perplexity for scheme={args.scheme}...")
    if args.quick:
        ppl = quick_perplexity(model, tokenizer, device=device)
        print(f"Quick perplexity estimate: {ppl:.2f}")
    else:
        ppl = compute_perplexity(model, tokenizer, device=device)
        print(f"Full WikiText-2 perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
