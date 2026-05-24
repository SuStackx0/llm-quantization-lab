"""
Perplexity Evaluation
=====================

Perplexity is THE standard metric for measuring language model quality.

What is perplexity?
-------------------
It measures how "surprised" the model is by real text.
- Low perplexity → model predicts text well → high quality
- High perplexity → model is confused by the text → lower quality

Mathematically: perplexity = exp(average negative log-likelihood)

For TinyLlama-1.1B:
  FP32 baseline:   ~7-8   (good)
  INT8 quantized:  ~7-9   (small degradation)
  INT4 naive:      ~10-15 (noticeable degradation)
  INT4 GPTQ:       ~7-10  (much better than naive INT4!)

We evaluate on WikiText-2, a standard dataset of Wikipedia articles.
"""

import torch
from datasets import load_dataset
from tqdm import tqdm
import math
from typing import Optional


def compute_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_length: int = 512,
    stride: int = 256,
    n_samples: Optional[int] = None,
    device: str = "cpu",
) -> float:
    """
    Compute WikiText-2 perplexity using a sliding window approach.

    The sliding window handles texts longer than the model's context length:
    we slide a window of size max_length across the text, overlapping by
    (max_length - stride) tokens each step. Only the new tokens in each
    window contribute to the loss.

    Args:
        model:          HuggingFace language model
        tokenizer:      Corresponding tokenizer
        dataset_name:   HuggingFace dataset name
        dataset_config: Dataset configuration
        split:          "test" for final evaluation, "validation" for quick checks
        max_length:     Maximum sequence length (context window size)
        stride:         How many new tokens per sliding step
        n_samples:      Limit evaluation to this many samples (None = full dataset)
        device:         Where to run inference

    Returns:
        perplexity score (lower = better)
    """
    print(f"Loading {dataset_name}/{dataset_config} ({split} split)...")
    dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)

    # Concatenate all text into one long string, then tokenize
    # WikiText-2 test set is ~2MB of text
    all_text = "\n\n".join(dataset["text"])
    encodings = tokenizer(all_text, return_tensors="pt")
    input_ids = encodings["input_ids"].squeeze()  # shape [total_tokens]

    # Optionally limit to n_samples tokens for faster evaluation
    if n_samples is not None:
        input_ids = input_ids[: n_samples * max_length]

    total_tokens = input_ids.shape[0]
    print(f"Total tokens: {total_tokens:,}")

    model.eval()
    total_log_likelihood = 0.0
    total_counted_tokens = 0

    # Sliding window over the full token sequence
    positions = range(0, total_tokens - 1, stride)
    for start in tqdm(positions, desc="Computing perplexity"):
        end = min(start + max_length, total_tokens)
        chunk = input_ids[start:end].unsqueeze(0).to(device)  # [1, chunk_len]

        with torch.no_grad():
            outputs = model(chunk, labels=chunk)
            log_likelihood = outputs.loss  # cross-entropy loss = -mean log P(token)

        # Only count the "new" tokens (not the overlapping prefix)
        # For the first window, all tokens are new
        # For subsequent windows, only stride tokens are new
        n_new_tokens = min(stride, end - start - 1)
        if start == 0:
            n_new_tokens = end - start - 1

        total_log_likelihood += log_likelihood.item() * n_new_tokens
        total_counted_tokens += n_new_tokens

        if end >= total_tokens:
            break

    avg_log_likelihood = total_log_likelihood / total_counted_tokens
    perplexity = math.exp(avg_log_likelihood)
    return perplexity


def quick_perplexity(model, tokenizer, n_tokens: int = 2048, device: str = "cpu") -> float:
    """
    Fast perplexity estimate using a small slice of WikiText-2.
    Use this during development — takes ~10 seconds instead of ~5 minutes.

    Args:
        n_tokens: How many tokens to evaluate on (smaller = faster but noisier)

    Returns:
        Approximate perplexity (not identical to full eval but useful for comparison)
    """
    return compute_perplexity(
        model, tokenizer,
        split="validation",
        max_length=256,
        stride=128,
        n_samples=n_tokens // 256 + 1,
        device=device,
    )
