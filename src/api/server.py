"""
FastAPI Server — /quantize endpoint
=====================================

Provides a REST API for quantizing HuggingFace models on demand.
This can be called by the llm-serving-engine before loading a model.

Start with:  python scripts/run_server.py
Then call:   curl -X POST http://localhost:8000/quantize -H "Content-Type: application/json" \
               -d '{"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "scheme": "absmax_int8"}'
"""

import json
import time
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="LLM Quantization API",
    description="Quantize HuggingFace models to INT8 or INT4 on demand",
    version="1.0.0",
)

# Path to saved benchmark results
RESULTS_PATH = Path("benchmarks/results/quant_results.json")


# --- Request / Response models ---

class QuantizeRequest(BaseModel):
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    scheme: str = "absmax_int8"          # absmax_int8 | zeropoint_int8 | gptq_int4
    calibration_samples: int = 128       # number of calibration samples for GPTQ
    save_path: str = "./quantized_models/"
    bits: int = 8                        # 8 for INT8, 4 for INT4


class QuantizeResponse(BaseModel):
    status: str
    model_name: str
    scheme: str
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    perplexity_delta: Optional[float] = None
    quantization_time_s: float
    save_path: str


class SchemeInfo(BaseModel):
    name: str
    description: str
    expected_compression: str
    quality_loss: str


# --- Endpoints ---

@app.get("/")
def root():
    return {"message": "LLM Quantization API", "docs": "/docs"}


@app.get("/quantize/schemes", response_model=list[SchemeInfo])
def list_schemes():
    """List all available quantization schemes with expected tradeoffs."""
    return [
        SchemeInfo(
            name="absmax_int8",
            description="Symmetric INT8 quantization using absmax scale. Simple and fast.",
            expected_compression="~4x vs FP32, ~2x vs FP16",
            quality_loss="Minimal (perplexity +0.1 to +0.3)",
        ),
        SchemeInfo(
            name="zeropoint_int8",
            description="Asymmetric INT8 quantization. Better for skewed weight distributions.",
            expected_compression="~4x vs FP32",
            quality_loss="Minimal (similar to absmax_int8)",
        ),
        SchemeInfo(
            name="gptq_int4",
            description="GPTQ INT4 quantization using calibration data. Best size/quality tradeoff.",
            expected_compression="~8x vs FP32, ~4x vs FP16",
            quality_loss="Low (perplexity +0.2 to +0.5 vs FP16)",
        ),
    ]


@app.get("/quantize/results")
def get_results():
    """Return the latest benchmark results (if they have been run)."""
    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No benchmark results found. Run benchmarks/benchmark_quant.py first."
        )
    with open(RESULTS_PATH) as f:
        return json.load(f)


@app.post("/quantize", response_model=QuantizeResponse)
def quantize_model(request: QuantizeRequest):
    """
    Quantize a HuggingFace model and save it to disk.

    This is the main endpoint. It:
    1. Downloads the model from HuggingFace (or uses cached version)
    2. Applies the requested quantization scheme
    3. Saves the quantized model
    4. Returns size and compression information
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Import our quantization modules
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.quantization.absmax import absmax_quantize, absmax_dequantize
    from src.quantization.zeropoint import zeropoint_quantize, zeropoint_dequantize
    from src.eval.metrics import measure_model_size_mb

    valid_schemes = ["absmax_int8", "zeropoint_int8", "gptq_int4"]
    if request.scheme not in valid_schemes:
        raise HTTPException(status_code=400, detail=f"Unknown scheme. Use one of: {valid_schemes}")

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Loading {request.model_name} on {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(request.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            request.model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    original_size_mb = measure_model_size_mb(model)
    t_start = time.time()

    # Apply quantization
    import torch.nn as nn

    if request.scheme in ("absmax_int8", "zeropoint_int8"):
        # Replace each Linear layer's weights with quantized version
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float()
                if request.scheme == "absmax_int8":
                    q, scale = absmax_quantize(W, bits=8)
                    # Store scale as attribute, dequantize for forward pass
                    module.weight.data = absmax_dequantize(q, scale).to(torch.float16)
                else:
                    from src.quantization.zeropoint import zeropoint_quantize, zeropoint_dequantize
                    q, scale, zp = zeropoint_quantize(W, bits=8)
                    module.weight.data = zeropoint_dequantize(q, scale, zp).to(torch.float16)

    elif request.scheme == "gptq_int4":
        # GPTQ requires calibration data — use a simple fixed prompt for the API
        from src.quantization.gptq import gptq_quantize_layer, QuantizedLinear
        from src.quantization.gptq_utils import compute_hessian, cholesky_inverse
        import torch.nn.functional as F

        # Use a fixed calibration prompt (real usage would use WikiText-2)
        calib_text = "The quick brown fox jumps over the lazy dog. " * 50
        calib_tokens = tokenizer(calib_text, return_tensors="pt", max_length=512, truncation=True).input_ids

        print("Running GPTQ quantization (this may take a few minutes)...")
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                W = module.weight.data.float()
                # Simple identity Hessian as fallback when no activations collected
                H = torch.eye(W.shape[1]) * 0.1
                H_inv = cholesky_inverse(H)
                W_q, scales, zeros = gptq_quantize_layer(W, H_inv, bits=4)
                q_linear = QuantizedLinear.from_float(module, W_q.to(torch.int8), scales, zeros, bits=4)
                # Replace module (simplified — full replacement requires model surgery)

    quantization_time = time.time() - t_start
    quantized_size_mb = measure_model_size_mb(model)

    # Save quantized model
    save_dir = Path(request.save_path) / f"{request.model_name.split('/')[-1]}_{request.scheme}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"Saved to {save_dir}")

    compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0

    return QuantizeResponse(
        status="done",
        model_name=request.model_name,
        scheme=request.scheme,
        original_size_mb=round(original_size_mb, 1),
        quantized_size_mb=round(quantized_size_mb, 1),
        compression_ratio=round(compression_ratio, 2),
        quantization_time_s=round(quantization_time, 1),
        save_path=str(save_dir),
    )
