"""
Quantization Metrics
====================

Utilities to measure everything we care about when comparing quantization schemes:
- Model size (how many MB on disk / in memory)
- Peak memory during inference
- Generation speed (tokens per second)
- Time to first token (TTFT)
"""

import time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QuantizationResult:
    """
    All metrics for one quantization scheme.
    Think of this as one row in our comparison table.
    """
    scheme: str                       # e.g. "FP32", "absmax_INT8", "GPTQ_INT4"
    model_size_mb: float = 0.0        # total parameter memory in megabytes
    perplexity: float = 0.0           # WikiText-2 perplexity (lower = better)
    tokens_per_second: float = 0.0    # generation throughput
    ttft_ms: float = 0.0              # time to first token in milliseconds
    peak_memory_mb: float = 0.0       # peak RAM during generation
    quantization_time_s: float = 0.0  # time to apply quantization (0 for prebuilt)

    def __str__(self):
        return (
            f"[{self.scheme}]\n"
            f"  Size:        {self.model_size_mb:.1f} MB\n"
            f"  Perplexity:  {self.perplexity:.2f}\n"
            f"  Speed:       {self.tokens_per_second:.1f} tok/s\n"
            f"  TTFT:        {self.ttft_ms:.1f} ms\n"
            f"  Peak Mem:    {self.peak_memory_mb:.1f} MB\n"
            f"  Quant Time:  {self.quantization_time_s:.1f} s"
        )


def measure_model_size_mb(model: nn.Module) -> float:
    """
    Measure total memory used by all parameters and buffers in the model.

    This counts the in-memory footprint, not the file size.
    For a quantized model, this reflects the actual integer storage.
    """
    total_bytes = 0
    for param in model.parameters():
        # element_size() = bytes per element (4 for float32, 2 for float16, 1 for int8)
        total_bytes += param.nelement() * param.element_size()
    for buf in model.buffers():
        total_bytes += buf.nelement() * buf.element_size()
    return total_bytes / (1024 ** 2)  # convert bytes to MB


def measure_peak_memory_mb(device: str) -> float:
    """
    Measure current peak memory allocation on the given device.

    For Apple Silicon (MPS):   uses torch.mps.driver_allocated_memory()
    For CUDA:                  uses torch.cuda.max_memory_allocated()
    For CPU:                   uses process RSS via psutil (if available)
    """
    if device == "mps":
        return torch.mps.driver_allocated_memory() / (1024 ** 2)
    elif device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        try:
            import psutil, os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 ** 2)
        except ImportError:
            return 0.0


def measure_generation_speed(
    model,
    tokenizer,
    prompt: str = "The quick brown fox",
    n_tokens: int = 64,
    n_runs: int = 3,
    device: str = "cpu",
) -> tuple[float, float]:
    """
    Measure tokens-per-second and time-to-first-token.

    Runs the model n_runs times and averages the results to reduce noise.

    Args:
        n_tokens: How many new tokens to generate per run
        n_runs:   Number of repetitions for averaging

    Returns:
        (tokens_per_second, ttft_ms)
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    ttft_list = []
    tps_list = []

    model.eval()
    with torch.no_grad():
        for _ in range(n_runs):
            # Measure TTFT: time until we get the first generated token
            t_start = time.perf_counter()
            outputs = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            t_first = time.perf_counter()
            ttft_list.append((t_first - t_start) * 1000)  # ms

            # Measure throughput: total time for n_tokens
            t_gen_start = time.perf_counter()
            outputs = model.generate(
                input_ids,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            t_gen_end = time.perf_counter()

            actual_new_tokens = outputs.shape[1] - input_ids.shape[1]
            elapsed = t_gen_end - t_gen_start
            tps_list.append(actual_new_tokens / elapsed if elapsed > 0 else 0)

    ttft = sum(ttft_list) / len(ttft_list)
    tps = sum(tps_list) / len(tps_list)
    return tps, ttft
