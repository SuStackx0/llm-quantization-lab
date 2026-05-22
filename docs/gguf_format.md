# GGUF Format: How llama.cpp Stores Models on Disk

> "GGUF is the file format that powers Ollama, llama.cpp, and LM Studio. Once you understand it, you understand the plumbing behind every local LLM deployment."

**Code:** `src/gguf/reader.py`, `src/gguf/k_quants.py`, `src/gguf/loader.py`

---

## What Is GGUF?

GGUF stands for **GGML Universal Format**. It replaced the older GGML format in 2023 to:
- Support richer metadata (model architecture, tokenizer, hyperparameters)
- Handle multiple quantization types in one file
- Be self-contained (one file = complete deployable model)

When you run `ollama run llama3`, Ollama downloads a `.gguf` file and passes it to llama.cpp for inference. The `.gguf` file contains:
- The model's quantized weights
- The tokenizer vocabulary
- All hyperparameters (context length, number of heads, etc.)

---

## File Structure Overview

A GGUF file is a binary file organized like this:

```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER (28 bytes)                                              │
│  ├── magic[4]:     "GGUF" (identifies file type)               │
│  ├── version[4]:   2 or 3 (format version)                     │
│  ├── n_tensors[8]: count of weight tensors                      │
│  └── n_kv[8]:      count of metadata key-value pairs           │
├─────────────────────────────────────────────────────────────────┤
│  METADATA (variable size)                                       │
│  n_kv entries of the form:                                     │
│  ├── key:   length(8) + UTF-8 string                           │
│  ├── type:  uint32 (0=uint8, 8=string, 9=array, ...)           │
│  └── value: depends on type                                     │
│                                                                 │
│  Examples:                                                      │
│  "general.name"        → "Llama-3.2-1B-Instruct"              │
│  "llama.block_count"   → 16 (number of transformer layers)     │
│  "llama.context_length" → 131072 (max tokens)                  │
│  "tokenizer.ggml.tokens" → ["<unk>", "<s>", ...]               │
├─────────────────────────────────────────────────────────────────┤
│  TENSOR INFO (variable size)                                    │
│  n_tensors entries of the form:                                 │
│  ├── name:      length(8) + UTF-8 string                        │
│  ├── n_dims:    uint32 (usually 1 or 2)                         │
│  ├── dims:      n_dims × uint64 (shape)                         │
│  ├── dtype:     uint32 (quantization type ID)                   │
│  └── offset:    uint64 (byte offset into tensor data section)  │
│                                                                 │
│  Note: shapes are stored "backwards" vs PyTorch convention     │
│  GGUF stores [fastest_dim, ..., slowest_dim]                   │
│  PyTorch wants [slowest_dim, ..., fastest_dim]                 │
├─────────────────────────────────────────────────────────────────┤
│  ALIGNMENT PADDING                                              │
│  Pads to 32-byte alignment for efficient memory mapping        │
├─────────────────────────────────────────────────────────────────┤
│  TENSOR DATA                                                    │
│  Raw bytes for each tensor, back-to-back                       │
│  Format of bytes depends on the tensor's dtype (Q4_K, Q8_0...) │
└─────────────────────────────────────────────────────────────────┘
```

---

## Reading the Header

Our reader (`GGUFReader.__init__`) does this in `_parse_header()`:

```python
magic = f.read(4)          # should be b"GGUF"
version = read_uint32(f)   # should be 2 or 3
n_tensors = read_uint64(f) # how many weight tensors
n_kv = read_uint64(f)      # how many metadata entries
```

We use Python's `struct` module for all binary reading:

```python
# "< " = little-endian (GGUF uses little-endian byte order)
# "B"  = unsigned byte (1 byte)
# "I"  = unsigned int (4 bytes)
# "Q"  = unsigned long long (8 bytes)
# "e"  = float16 (2 bytes)
# "f"  = float32 (4 bytes)

value = struct.unpack("<I", f.read(4))[0]   # read 4 bytes as uint32
```

---

## Metadata Key-Value Pairs

Each metadata entry has:
1. Key: a string (length as uint64, then UTF-8 bytes)
2. Type: uint32 indicating the value's type
3. Value: depends on type

```
Type 0  = uint8    (1 byte)
Type 4  = uint32   (4 bytes)
Type 6  = float32  (4 bytes)
Type 8  = string   (length:uint64 + utf8 bytes)
Type 9  = array    (elem_type:uint32 + count:uint64 + elements)
Type 10 = uint64   (8 bytes)
```

Common metadata keys for LLaMA models:
```
general.name           → model name string
general.architecture   → "llama"
llama.block_count      → number of transformer layers (e.g. 16 for 1B, 32 for 8B)
llama.context_length   → maximum sequence length
llama.embedding_length → hidden dimension size (d_model)
llama.attention.head_count → number of attention heads
llama.rope.freq_base   → RoPE frequency base
tokenizer.ggml.model   → "llama" (tokenizer type)
tokenizer.ggml.tokens  → array of vocabulary token strings
tokenizer.ggml.scores  → array of token scores/frequencies
```

---

## Tensor Info Section

After metadata, we read `n_tensors` tensor descriptors:

```python
name = read_string(f)      # e.g. "blk.0.attn_q.weight"
n_dims = read_uint32(f)    # usually 1 (vectors) or 2 (matrices)
dims = [read_uint64(f) for _ in range(n_dims)]   # e.g. [4096, 4096]
dtype_id = read_uint32(f)  # e.g. 13 = Q4_K_M
offset = read_uint64(f)    # byte offset into tensor data section
```

**Important:** The `dims` are in GGUF order (fastest-varying first), which is the reverse of PyTorch. Our reader reverses them:

```python
shape = tuple(reversed(dims))  # convert to standard PyTorch [out, in] order
```

### Quantization Type IDs

| ID | Name | Description |
|----|------|-------------|
| 0 | F32 | Full precision float32 |
| 1 | F16 | Half precision float16 |
| 8 | Q8_0 | 8-bit, 1 scale per 32 weights |
| 12 | Q4_K_S | 4-bit small k-quant |
| 13 | Q4_K_M | 4-bit medium k-quant (most common) |
| 14 | Q5_K_S | 5-bit small k-quant |
| 15 | Q5_K_M | 5-bit medium k-quant |
| 16 | Q6_K | 6-bit k-quant |

---

## Tensor Data: How Q8_0 Is Encoded on Disk

Q8_0 is the simplest quantization format — a great starting point.

### Q8_0 Block Layout (34 bytes per 32 weights)

```
Byte offset: [  0  1 ] [  2  3  4  5  ...  33 ]
Content:      [ d fp16] [  q0 q1 q2 ... q31    ]

d   = float16 scale factor (2 bytes)
q0..q31 = int8 quantized values (32 bytes)
```

**Dequantization:** `w[i] = d × q[i]`

To read this in Python:
```python
import struct
import numpy as np

# Read one Q8_0 block
block = raw_bytes[offset : offset + 34]
d = struct.unpack_from("<e", block, 0)[0]           # float16 at offset 0
qs = np.frombuffer(block, dtype=np.int8, count=32, offset=2)  # 32 int8 at offset 2
weights = d * qs.astype(np.float32)                 # dequantize
```

---

## Tensor Data: How Q4_K Is Encoded on Disk

Q4_K_M is the most popular format. Harder to parse but better quality.

### Q4_K Super-Block Layout (144 bytes per 256 weights)

```
Bytes 0-1:    d     (fp16 super-block scale)
Bytes 2-3:    dmin  (fp16 super-block min)
Bytes 4-15:   12 bytes of packed 6-bit scales and mins
              (8 sub-block scales + 8 sub-block mins = 16 values × 6 bits = 96 bits = 12 bytes)
Bytes 16-143: 128 bytes = 256 × 4-bit quantized values (2 values packed per byte)
```

### Unpacking 4-Bit Values

256 values packed into 128 bytes, two per byte:

```python
qs = np.frombuffer(block, dtype=np.uint8, count=128, offset=16)  # [128]
q_low  = qs & 0x0F          # lower 4 bits → values 0-127  (first 128 weights)
q_high = (qs >> 4) & 0x0F   # upper 4 bits → values 128-255 (second 128 weights)
q_all  = np.concatenate([q_low, q_high])   # [256] values, each 0..15
```

### Decoding 6-Bit Scales

The 12 bytes encode 16 six-bit values (8 sub-scales + 8 sub-mins):

```python
# Bytes 0..5: lower 6 bits are sub_scale[0..5]
# Bytes 6..11: lower 6 bits are sub_min[0..5]
# Upper 2 bits of bytes 0..11 encode sub_scale[6..7] and sub_min[6..7]

for i in range(6):
    sub_scales[i] = b[i] & 0x3F      # lower 6 bits
for i in range(6):
    sub_mins[i] = b[i+6] & 0x3F

# Last two values assembled from upper bits
sub_scales[6] = (b[0] >> 6) | ((b[1] >> 6) << 2) | ((b[2] >> 6) << 4)
sub_scales[7] = (b[3] >> 6) | ((b[4] >> 6) << 2) | ((b[5] >> 6) << 4)
```

### Full Q4_K Dequantization

```python
for sub in range(8):          # 8 sub-blocks of 32 weights each
    scale = d * sub_scales[sub]
    mn    = dmin * sub_mins[sub]
    weights[sub*32 : (sub+1)*32] = scale * q_all[sub*32 : (sub+1)*32] - mn
```

---

## GGUF Tensor Names vs HuggingFace Names

GGUF uses a compact naming scheme; HuggingFace uses a verbose one:

| GGUF Name | HuggingFace Name |
|-----------|-----------------|
| `token_embd.weight` | `model.embed_tokens.weight` |
| `blk.0.attn_q.weight` | `model.layers.0.self_attn.q_proj.weight` |
| `blk.0.attn_k.weight` | `model.layers.0.self_attn.k_proj.weight` |
| `blk.0.attn_v.weight` | `model.layers.0.self_attn.v_proj.weight` |
| `blk.0.attn_output.weight` | `model.layers.0.self_attn.o_proj.weight` |
| `blk.0.ffn_gate.weight` | `model.layers.0.mlp.gate_proj.weight` |
| `blk.0.ffn_up.weight` | `model.layers.0.mlp.up_proj.weight` |
| `blk.0.ffn_down.weight` | `model.layers.0.mlp.down_proj.weight` |
| `blk.0.attn_norm.weight` | `model.layers.0.input_layernorm.weight` |
| `output_norm.weight` | `model.norm.weight` |
| `output.weight` | `lm_head.weight` |

The mapping is applied in `src/gguf/loader.py` using the `GGUF_TO_HF` dictionary.

---

## How This Connects to Ollama / llama.cpp

When you run `ollama run llama3:8b`:

1. Ollama downloads `llama3:8b-q4_K_M.gguf` (or uses cached)
2. Passes the path to llama.cpp's `llama_model_load()`
3. llama.cpp `mmap()`s the file (maps it to memory without copying)
4. Builds internal tensor metadata from the GGUF tensor info section
5. For each forward pass token, reads the needed tensor bytes on-demand
6. Dequantizes Q4_K super-blocks during the matmul

Our `GGUFReader` does steps 2-4 in Python. The `load_tensor()` method does step 5-6 for any tensor you name.

**Key advantage of memory-mapping:** The OS manages which pages of the file are in RAM. For a 4GB GGUF model, if you only access 2GB of weights during inference, the OS keeps the other 2GB on disk until needed. This is why llama.cpp can "run" a 70B model on a machine with only 32GB RAM (slowly, but it works).
