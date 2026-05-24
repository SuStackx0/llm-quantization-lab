"""
Tests for GGUF Reader
=======================

Run with:  pytest tests/test_gguf_reader.py -v

Most tests use synthetic/fake GGUF bytes to test parsing logic
without needing to download a real GGUF file.
The real-file test is skipped if no .gguf file is present.
"""

import struct
import torch
import pytest
import tempfile
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gguf.k_quants import dequantize_q8_0, dequantize_q4_k


# ---- k_quants dequantization tests ----

def make_q8_0_block(scale: float, values: list[int]) -> bytes:
    """Helper: create a Q8_0 block with known scale and values."""
    assert len(values) == 32, "Q8_0 block must have exactly 32 values"
    data = struct.pack("<e", scale)   # 2 bytes: float16 scale
    data += struct.pack("<32b", *values)  # 32 bytes: int8 values
    return data


def test_q8_0_simple():
    """Q8_0: d=1.0, values=[1,2,...,32] should dequantize to [1,2,...,32]."""
    values = list(range(1, 33))
    block = make_q8_0_block(1.0, values)
    result = dequantize_q8_0(block, (32,))
    expected = torch.tensor(values, dtype=torch.float32)
    assert torch.allclose(result, expected, atol=0.01), \
        f"Q8_0 dequant wrong. Expected {expected[:5]}, got {result[:5]}"


def test_q8_0_scale():
    """Q8_0: d=0.5 should halve all values."""
    values = [10] * 32
    block = make_q8_0_block(0.5, values)
    result = dequantize_q8_0(block, (32,))
    assert torch.allclose(result, torch.full((32,), 5.0), atol=0.1), \
        f"Q8_0 with scale 0.5 should give 5.0, got {result[0]}"


def test_q8_0_negative_values():
    """Q8_0 handles negative int8 values."""
    values = [-10] * 32
    block = make_q8_0_block(2.0, values)
    result = dequantize_q8_0(block, (32,))
    assert torch.allclose(result, torch.full((32,), -20.0), atol=0.5)


def test_q8_0_shape():
    """Q8_0 output shape matches requested shape."""
    values = list(range(32))
    block = make_q8_0_block(1.0, values)
    result = dequantize_q8_0(block, (32,))
    assert result.shape == (32,)


def test_q8_0_multiple_blocks():
    """Q8_0 with two blocks."""
    block1 = make_q8_0_block(1.0, [1] * 32)
    block2 = make_q8_0_block(2.0, [3] * 32)
    data = block1 + block2
    result = dequantize_q8_0(data, (64,))
    # First block: 1.0 * 1 = 1.0
    assert torch.allclose(result[:32], torch.ones(32), atol=0.01)
    # Second block: 2.0 * 3 = 6.0
    assert torch.allclose(result[32:], torch.full((32,), 6.0), atol=0.1)


def test_q4_k_shape():
    """Q4_K dequantization produces correct shape."""
    # Create a minimal synthetic Q4_K block (144 bytes = 1 super-block of 256 weights)
    block = bytearray(144)
    # Set d=1.0, dmin=0.0 (fp16)
    struct.pack_into("<e", block, 0, 1.0)
    struct.pack_into("<e", block, 2, 0.0)
    # scale bytes 4..15: set all sub-scales to 8 (maps to 8.0 / 63 * 1.0 ≈ 0.127)
    for i in range(4, 16):
        block[i] = 8

    result = dequantize_q4_k(bytes(block), (256,))
    assert result.shape == (256,), f"Expected shape (256,), got {result.shape}"
    assert result.dtype == torch.float32


def test_q4_k_zero_block():
    """Q4_K with all zeros should produce all-zero weights."""
    block = bytes(144)   # all zero bytes
    result = dequantize_q4_k(block, (256,))
    # d=0 means all weights = 0
    assert result.abs().max().item() < 1e-6


# ---- GGUF reader integration test (skipped if no .gguf file) ----

def find_gguf_file():
    """Look for any .gguf file in common locations."""
    search_paths = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/models"),
        ".",
    ]
    for path in search_paths:
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(".gguf"):
                    return os.path.join(root, f)
    return None


@pytest.mark.skipif(find_gguf_file() is None, reason="No .gguf file found locally")
def test_real_gguf_reader():
    """Integration test: read a real GGUF file if available."""
    from src.gguf.reader import GGUFReader
    gguf_path = find_gguf_file()
    print(f"\nTesting with real GGUF file: {gguf_path}")

    reader = GGUFReader(gguf_path)

    # Basic structural checks
    assert len(reader.tensor_infos) > 0, "Should have at least one tensor"
    assert len(reader.metadata) > 0, "Should have metadata"

    # Version should be 2 or 3
    assert reader.version in [2, 3], f"Unexpected GGUF version: {reader.version}"

    # First tensor should be loadable
    first_tensor_name = reader.tensor_infos[0].name
    if reader.tensor_infos[0].dtype_id in [0, 1, 8, 12, 13, 14, 15]:
        tensor = reader.load_tensor(first_tensor_name)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        print(f"Loaded tensor '{first_tensor_name}': shape={tensor.shape}")
