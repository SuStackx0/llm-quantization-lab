"""
CLI Script: Quantize a HuggingFace Model
==========================================

Usage:
    python scripts/quantize_model.py --scheme absmax_int8
    python scripts/quantize_model.py --scheme zeropoint_int8
    python scripts/quantize_model.py --scheme gptq_int4 --calibration_samples 128

This script:
1. Loads TinyLlama-1.1B (or any HF model)
2. Applies the chosen quantization scheme
3. Saves the quantized model to disk
4. Prints before/after memory and a sample generation
"""

import sys
import os
import argparse
import time
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.quantization.absmax import absmax_quantize, absmax_dequantize
from src.quantization.zeropoint import zeropoint_quantize, zeropoint_dequantize
from src.quantization.gptq import gptq_quantize_layer, QuantizedLinear
from src.quantization.gptq_utils import collect_input_stats, compute_hessian, cholesky_inverse
from src.eval.metrics import measure_model_size_mb, measure_generation_speed


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def apply_absmax_int8(model):
    """Replace all nn.Linear weights with absmax INT8 quantized versions."""
    print("\nApplying absmax INT8 quantization...")
    n_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data.float()
            q, scale = absmax_quantize(W, bits=8)
            # Dequantize back so the model can still run (weight-storage quantization)
            # In a real deployment you'd store int8 and dequantize just-in-time
            module.weight.data = absmax_dequantize(q, scale).to(module.weight.dtype)
            n_layers += 1
    print(f"  Quantized {n_layers} linear layers")
    return model


def apply_zeropoint_int8(model):
    """Replace all nn.Linear weights with zero-point INT8 quantized versions."""
    print("\nApplying zero-point INT8 quantization...")
    n_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data.float()
            q, scale, zp = zeropoint_quantize(W, bits=8)
            module.weight.data = zeropoint_dequantize(q, scale, zp).to(module.weight.dtype)
            n_layers += 1
    print(f"  Quantized {n_layers} linear layers")
    return model


def apply_gptq_int4(model, tokenizer, n_calib_samples: int = 128, device: str = "cpu"):
    """
    Apply GPTQ INT4 quantization layer by layer.

    For each Linear layer:
    1. Collect input activations via forward hook
    2. Compute Hessian and its inverse
    3. Run GPTQ to get INT4 weights with error compensation
    4. Replace layer with QuantizedLinear
    """
    print(f"\nApplying GPTQ INT4 quantization (n_calib={n_calib_samples})...")

    # Load calibration data from WikiText-2
    print("  Loading WikiText-2 calibration data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    texts = [t for t in dataset["text"] if len(t.strip()) > 50][:n_calib_samples]

    # Tokenize calibration samples
    calib_tokens = tokenizer(
        texts,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).input_ids.to(device)

    model.eval()
    replacements = {}  # {name: new_module}

    # Get list of all Linear layer names
    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and "lm_head" not in name
    ]

    print(f"  Found {len(linear_layers)} linear layers to quantize")

    for layer_idx, (name, module) in enumerate(linear_layers):
        print(f"  [{layer_idx+1}/{len(linear_layers)}] Quantizing {name}...", end=" ", flush=True)
        t0 = time.time()

        try:
            # Collect activations for this layer
            X = collect_input_stats(model, calib_tokens, name, device=device)

            # Compute Hessian and inverse
            H = compute_hessian(X, damp=0.01)
            H_inv = cholesky_inverse(H)

            # Run GPTQ
            W = module.weight.data.float()
            W_q, scales, zeros = gptq_quantize_layer(W, H_inv, bits=4)

            # Create QuantizedLinear replacement
            q_linear = QuantizedLinear.from_float(module, W_q.to(torch.int8), scales, zeros, bits=4)
            replacements[name] = q_linear

            print(f"done ({time.time()-t0:.1f}s)")

        except Exception as e:
            print(f"FAILED ({e}) — keeping original FP16")

    # Apply replacements (two-pass to avoid modifying dict during iteration)
    for name, new_module in replacements.items():
        # Navigate to the parent module and replace
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    print(f"\n  Replaced {len(replacements)}/{len(linear_layers)} layers with QuantizedLinear")
    return model


def main():
    parser = argparse.ArgumentParser(description="Quantize a HuggingFace LLM")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--scheme", choices=["absmax_int8", "zeropoint_int8", "gptq_int4"],
                        default="absmax_int8")
    parser.add_argument("--calibration_samples", type=int, default=128)
    parser.add_argument("--save_path", default="./quantized_models/")
    parser.add_argument("--no_save", action="store_true", help="Skip saving to disk")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=device,
    )

    original_size = measure_model_size_mb(model)
    print(f"Original model size: {original_size:.1f} MB")

    # Sample generation BEFORE quantization
    print("\n--- Generation BEFORE quantization ---")
    prompt = "Explain what machine learning is in one sentence:"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    # Apply quantization
    t_start = time.time()
    if args.scheme == "absmax_int8":
        model = apply_absmax_int8(model)
    elif args.scheme == "zeropoint_int8":
        model = apply_zeropoint_int8(model)
    elif args.scheme == "gptq_int4":
        model = apply_gptq_int4(model, tokenizer, args.calibration_samples, device)
    quant_time = time.time() - t_start

    quantized_size = measure_model_size_mb(model)
    compression = original_size / quantized_size if quantized_size > 0 else 1.0

    print(f"\n--- Results ---")
    print(f"Original size:   {original_size:.1f} MB")
    print(f"Quantized size:  {quantized_size:.1f} MB")
    print(f"Compression:     {compression:.2f}x")
    print(f"Quant time:      {quant_time:.1f}s")

    # Sample generation AFTER quantization
    print("\n--- Generation AFTER quantization ---")
    try:
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(out[0], skip_special_tokens=True))
    except Exception as e:
        print(f"Generation failed: {e}")

    # Save
    if not args.no_save:
        save_dir = os.path.join(args.save_path, f"{args.model.split('/')[-1]}_{args.scheme}")
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        print(f"\nSaved tokenizer to {save_dir}")
        print("(Skipping model save for QuantizedLinear — use torch.save(model.state_dict()) instead)")


if __name__ == "__main__":
    main()
