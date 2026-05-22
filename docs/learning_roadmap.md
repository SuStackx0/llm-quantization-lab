# Learning Roadmap — Concepts First, Code Second

> This document tells you **exactly what to know before opening each file**, and **exactly what each file will teach you**. Follow it top to bottom and nothing will feel unexplained.

---

## Part 0 — Concepts the Entire Project Needs

Before touching any code, make sure you are comfortable with these. They appear everywhere.

### Python Basics
- **Functions and return values** — every quantizer returns a tuple `(quantized_tensor, scale)`
- **Classes and `self`** — `QuantizedLinear`, `GGUFReader`, `QuantizationResult` are all classes
- **`import` and module paths** — `from src.quantization.absmax import absmax_quantize`
- **List comprehensions** — `[t.name for t in reader.tensor_infos]`
- **`with open(path, "rb") as f`** — binary file reading used in the GGUF parser

### PyTorch / Tensor Basics
- **What a tensor is** — a multi-dimensional array. `shape`, `dtype`, `nelement()`, `element_size()`
- **dtypes** — `torch.float32` (4 bytes), `torch.float16` (2 bytes), `torch.int8` (1 byte), `torch.uint8` (1 byte unsigned)
- **`torch.round()`, `.clamp()`, `.abs()`** — used in every quantizer
- **`tensor.float()`** — converts any integer tensor to float32 for math
- **`tensor.reshape()`, `tensor.unsqueeze()`** — shape manipulation
- **`torch.no_grad()`** — disables gradient tracking during inference (always used in eval)

### Neural Network Basics
- **What a weight matrix is** — `nn.Linear` has a `.weight` attribute of shape `[out_features, in_features]`
- **Forward pass** — input flows through layers: `y = W @ x + b`
- **`model.named_modules()`** — iterates over every layer in a model by name
- **`model.eval()`** — puts model in inference mode (no dropout, no batch norm updates)

### Language Model Basics
- **What TinyLlama is** — a small (~1.1B parameter) open-source language model
- **Tokenization** — text → integer token IDs via `tokenizer(text, return_tensors="pt")`
- **`model.generate()`** — produces new tokens given input tokens
- **HuggingFace `AutoModelForCausalLM`** — loads any causal LM from a name string

### The Core Quantization Idea (read before any code)
Quantization replaces 4-byte floats with 1-byte (or 0.5-byte) integers to save memory.
You need **two operations**:
```
quantize:   q = round(x / scale)         # float → integer
dequantize: x̂ = q × scale               # integer → approximate float
```
The difference `x - x̂` is **quantization error**. The whole project is about minimizing it.

---

## Part 1 — Absmax Quantization (The Simplest Scheme)

---

### Before reading `src/quantization/absmax.py`:

| Concept | Why you need it |
|---------|----------------|
| `tensor.abs().max()` | Computing the max absolute value is the whole point of "absmax" |
| `torch.round()` | Rounding floats to nearest integer |
| `.clamp(min, max)` | Preventing integer overflow after rounding |
| `.to(torch.int8)` | Storing the result in 1 byte instead of 4 |
| Powers of 2: `2^(bits-1) - 1` | For 8 bits: 127. For 4 bits: 7. This is the integer range. |

### `src/quantization/absmax.py`

### Concepts it teaches:
- **Scale factor** — `scale = max(|W|) / 127`. One number that encodes the entire precision budget.
- **Symmetric quantization** — the integer range `[-127, 127]` is symmetric around zero, just like most weight distributions.
- **INT8 vs INT4** — 8 bits gives 127 levels per side; 4 bits gives only 7. INT4 is 18× coarser.
- **Per-tensor vs per-channel** — one global scale vs one scale per row. Per-channel is always more accurate because rows have different magnitudes.
- **Memory formula** — `n_bytes = n_elements × element_size`. INT8 is 4× smaller than FP32.

---

### Before reading `src/quantization/zeropoint.py`:

| Concept | Why you need it |
|---------|----------------|
| `tensor.min()`, `tensor.max()` | Zero-point maps the actual `[min, max]` range |
| `torch.uint8` | Zero-point stores values in `[0, 255]` (unsigned) |
| What "asymmetric" means | The positive and negative sides get unequal space |

### `src/quantization/zeropoint.py`

### Concepts it teaches:
- **Zero-point offset** — an integer shift so that 0.0 maps to a specific integer. Allows non-zero-centered data to use the full `[0, 255]` range.
- **Why zero-point can be negative** — if all data is positive (e.g. `[1, 2, 3]`), then 0.0 would sit at a negative index. Zero-point is NOT clamped — it's a shift parameter stored as a plain int.
- **When asymmetric wins** — activations after ReLU are always positive. Absmax wastes half the range on the negative side that doesn't exist. Zero-point uses the full range.
- **Formula**: `scale = (max - min) / 255`, `zero_point = round(-min / scale)`, then `q = round(x / scale + zero_point)`

---

### Before reading `src/quantization/dequantize.py`:

| Concept | Why you need it |
|---------|----------------|
| Both absmax and zeropoint schemes | This file wraps both into one interface |
| `isinstance(scale, torch.Tensor)` | Detects whether we have per-channel (tensor) or per-tensor (float) scale |

### `src/quantization/dequantize.py`

### Concepts it teaches:
- **Unified interface pattern** — one function that dispatches based on scheme type instead of calling scheme-specific functions everywhere.
- **Broadcasting** — `scale` of shape `[out, 1]` multiplied by `quantized` of shape `[out, in]` automatically scales each row. PyTorch broadcasts the shapes.

---

## Part 2 — Testing Quantization Correctness

---

### Before reading `tests/test_absmax.py` and `tests/test_zeropoint.py`:

| Concept | Why you need it |
|---------|----------------|
| `pytest` basics — `def test_*()`, `assert` | How tests are written and run |
| `torch.allclose(a, b, atol=tol)` | Checks two tensors are numerically close (not identical) |
| Round-trip testing concept | The key invariant: `dequantize(quantize(x)) ≈ x` |
| Edge cases — zeros, single value, extreme ranges | What makes a quantizer fail in practice |

### `tests/test_absmax.py` and `tests/test_zeropoint.py`

### Concepts they teach:
- **Round-trip tolerance** — the acceptable error is `scale` (one quantization step). Anything beyond that is a bug.
- **Range assertions** — INT8 values must be in `[-127, 127]`. INT4 in `[-7, 7]`. Violations mean the clamp is wrong.
- **Edge case thinking** — all-zeros tensor (scale = 0 edge case), single-value (shape preservation), negative-only (sign handling).
- **Measurement of quantization error** — `(x - dequant(quant(x))).abs().mean()` is the key metric. Expect `< 1%` relative error for INT8.
- **Per-channel scales have a shape** — `[out_features, 1]`, not a flat list.

---

## Part 3 — Measuring Model Quality and Speed

---

### Before reading `src/eval/metrics.py`:

| Concept | Why you need it |
|---------|----------------|
| `model.parameters()` and `model.buffers()` | Where all stored tensors live in an nn.Module |
| `param.nelement() * param.element_size()` | How to compute bytes from a tensor |
| `time.perf_counter()` | High-precision timing in Python |
| `model.generate()` | Needed to measure generation speed |

### `src/eval/metrics.py`

### Concepts it teaches:
- **`@dataclass`** — a Python decorator that auto-generates `__init__` and other methods from field annotations. `QuantizationResult` is a dataclass.
- **Model size = sum of parameter bytes** — not file size on disk. Different schemes store different dtypes so size differs.
- **Tokens per second (TPS)** — `new_tokens / elapsed_seconds`. The main throughput metric.
- **Time to first token (TTFT)** — time until the first generated token appears. Measures latency, not throughput.
- **Peak memory** — `torch.mps.driver_allocated_memory()` on Apple Silicon, `torch.cuda.max_memory_allocated()` on GPU.

---

### Before reading `src/eval/perplexity.py`:

| Concept | Why you need it |
|---------|----------------|
| **Cross-entropy loss** — `model(tokens, labels=tokens).loss` | HuggingFace returns this directly |
| `math.exp()` | Perplexity = `exp(average cross-entropy)` |
| HuggingFace `datasets` — `load_dataset("wikitext", ...)` | The calibration/test dataset |
| Tokenization — `tokenizer(text, return_tensors="pt")` | Converting text to token IDs |
| **Why sliding window?** | Language models have a max context length. Long texts need to be chunked. |

### `src/eval/perplexity.py`

### Concepts it teaches:
- **Perplexity formula** — `exp(mean negative log-likelihood)`. Lower = better. `e^0 = 1` (perfect model), typical LLMs ~7-8 on WikiText-2.
- **WikiText-2** — a standard benchmark dataset of Wikipedia articles used to compare models since 2016.
- **Sliding window evaluation** — to handle texts longer than the context window, we slide a window of `max_length` tokens with `stride` overlap. Only the non-overlapping portion contributes to the loss.
- **Why perplexity degrades with quantization** — quantization introduces weight error → output distribution shifts → model assigns lower probability to correct tokens → higher perplexity.

---

## Part 4 — GPTQ (The Smart INT4 Algorithm)

---

### Before reading `src/quantization/gptq_utils.py`:

| Concept | Why you need it |
|---------|----------------|
| **What activations are** — the intermediate values flowing through a layer | GPTQ needs to observe these during calibration |
| **Forward hooks in PyTorch** — `module.register_forward_hook(fn)` | How to spy on what a layer receives as input |
| **Matrix multiplication review** — `X @ X.T` | How the Hessian is computed |
| **What a Hessian matrix is** | Second-order derivative: how sensitive the output is to each weight |
| **Why matrix inversion is hard numerically** | Near-singular matrices blow up. Cholesky avoids this. |

### `src/quantization/gptq_utils.py`

### Concepts it teaches:
- **Calibration data** — a small set of real text samples (128 sentences from WikiText-2) used to estimate weight importance. No gradient updates — just forward passes to collect statistics.
- **Forward hooks** — `module.register_forward_hook(callback)` lets you intercept the input to any layer during a forward pass. Always call `hook.remove()` after.
- **Hessian in this context** — `H = 2 * X @ X.T / n_tokens`. It's a `[d_in, d_in]` matrix. `H[i,i]` tells you: "how much does the output change if I perturb input dimension i?"
- **Damping** — adding `λ * mean(diag(H)) * I` to H prevents near-zero eigenvalues from making `H_inv` explode. Small λ (0.01) works well in practice.
- **Cholesky decomposition** — `H = L @ L.T` where L is lower-triangular. Inverting triangular matrices is numerically stable. `cholesky_inverse(L)` gives `H_inv`.

---

### Before reading `src/quantization/gptq.py`:

| Concept | Why you need it |
|---------|----------------|
| `gptq_utils.py` concepts (above) | GPTQ builds directly on them |
| **`torch.outer(a, b)`** — outer product of two vectors | The error propagation step uses this |
| **Column-by-column processing** — iterating `for j in range(d_in)` | GPTQ processes one column at a time |
| **Bit packing** — storing two 4-bit values per byte using bitwise AND `&` and shift `<<` | How `QuantizedLinear` cuts memory in half |
| **`nn.Module` and `register_buffer`** | Buffers are tensors saved in `state_dict` but not trained |

### `src/quantization/gptq.py`

### Concepts it teaches:
- **The GPTQ loop** — quantize column j → compute rounding error `e` → propagate `e` to remaining columns via `W[:, j+1:] -= outer(e, H_inv[j, j+1:]) / H_inv[j, j]`. The key insight: quantization errors can be partially cancelled by adjusting future weights.
- **Error compensation** — unlike naive rounding (which ignores error), GPTQ redistributes the rounding error to weights that can absorb it with less impact on output quality.
- **`QuantizedLinear`** — drops `nn.Linear`'s fp16 weight and instead stores `weight_packed` (uint8 with two INT4 values per byte), `scales` (float32), and `zeros` (float32). 4× smaller than FP32, 2× smaller than FP16.
- **INT4 packing/unpacking** — pack: `byte = (w0 & 0x0F) | ((w1 & 0x0F) << 4)`. Unpack: `low = byte & 0x0F`, `high = (byte >> 4) & 0x0F`. Two values per byte.
- **Weight-only quantization** — weights are stored as INT4 but dequantized to fp16/fp32 just before the matmul. The matmul itself still runs in fp16. Activations are never quantized.

---

### Before reading `tests/test_gptq.py`:

| Concept | Why you need it |
|---------|----------------|
| GPTQ concepts (above) | These tests verify those exact properties |
| **What symmetric positive definite means** | A Hessian should be SPD for Cholesky to work |
| **`H @ H_inv ≈ I`** | The correctness check for matrix inversion |

### `tests/test_gptq.py`

### Concepts it teaches:
- **Hessian symmetry as a sanity check** — `H == H.T` within float tolerance. If your Hessian is not symmetric, the computation is wrong.
- **Proving GPTQ is better than naive** — `test_gptq_better_than_naive` directly measures `(W - W_dequant).abs().mean()` for both methods and asserts GPTQ ≤ naive. This is the empirical proof of the algorithm.
- **Pack → unpack correctness** — `test_quantized_linear_pack_unpack` verifies INT4 bit packing preserves values. If >95% match, packing is correct.
- **Shape preservation** — quantizing a layer should never change the output shape. `test_quantized_linear_forward_shape` verifies this.

---

## Part 5 — GGUF Binary Format (How llama.cpp Stores Models)

---

### Before reading `src/gguf/k_quants.py`:

| Concept | Why you need it |
|---------|----------------|
| **Python `struct` module** — `struct.unpack_from("<e", data, offset)` | Reading typed values from raw bytes |
| **`numpy.frombuffer(data, dtype=np.int8, count=32, offset=2)`** | Interpreting a byte slice as a typed array |
| **Bit shifting** — `byte >> 4` (move upper nibble down), `byte & 0x0F` (mask lower nibble) | How 4-bit values are packed 2-per-byte |
| **Float16 (`"<e"` format code)** — half-precision scale factors | GGUF stores scales as fp16 |
| **What a "block" is** — a fixed group of weights with shared metadata | Q8_0 groups 32 weights per block; Q4_K groups 256 |

### `src/gguf/k_quants.py`

### Concepts it teaches:
- **Q8_0 block layout** — 34 bytes = 2 bytes (fp16 scale `d`) + 32 bytes (int8 values). Dequant: `w = d × q`. Simplest possible format.
- **Q4_K super-block layout** — 144 bytes = 2 (d) + 2 (dmin) + 12 (6-bit sub-scales and sub-mins for 8 sub-blocks) + 128 (256 × 4-bit values). Dequant: `w = d × sub_scale × q - dmin × sub_min`.
- **Why k-quants beat uniform INT4** — each 32-weight sub-block has its own adapted scale, so outliers in one sub-block don't ruin precision for all other sub-blocks.
- **6-bit packing** — 6-bit integers stored by combining lower 6 bits of one byte with upper 2 bits of another. Requires careful bit manipulation to decode.
- **Nibble unpacking** — 256 values stored in 128 bytes. Low nibble = first 128 values, high nibble = second 128 values. Concatenate after unpacking.

---

### Before reading `src/gguf/reader.py`:

| Concept | Why you need it |
|---------|----------------|
| `k_quants.py` (above) | The reader calls these to dequantize |
| **`open(path, "rb")` + `f.seek(offset)`** | Binary random access — reading bytes at a specific position |
| **`struct.unpack("<I", f.read(4))`** — little-endian uint32 | GGUF uses little-endian byte order throughout |
| **`@dataclass`** | `TensorInfo` is a dataclass |
| **Byte alignment** — rounding to the next multiple of 32 | GGUF aligns tensor data to 32 bytes |
| **What a "magic number" is** — first bytes of a file that identify its format | GGUF files start with the bytes `GGUF` |

### `src/gguf/reader.py`

### Concepts it teaches:
- **GGUF file structure** — 4 sections in order: header (28 bytes) → metadata (n_kv key-value pairs) → tensor info (n_tensors descriptors) → tensor data (raw bytes, back-to-back).
- **Lazy loading** — the reader parses metadata and tensor *info* (names, shapes, offsets) without loading tensor *data*. `load_tensor(name)` reads only the bytes for the requested tensor. This lets you inspect a 4GB file in milliseconds.
- **Variable-length binary format** — strings are encoded as (uint64 length) + (UTF-8 bytes). The format is not fixed-width, so you must read sequentially.
- **Type dispatch** — a `dict` maps type IDs (0=uint8, 8=string, 9=array) to reader functions. Calling `readers[type_id](f)` reads and returns the correct Python type.
- **Dimension reversal** — GGUF stores dims in [fastest-varying, ..., slowest-varying] order (C order). PyTorch uses [slowest, ..., fastest] (row-major). Reader reverses with `tuple(reversed(dims))`.

---

### Before reading `src/gguf/loader.py`:

| Concept | Why you need it |
|---------|----------------|
| `reader.py` (above) | Loader wraps reader |
| **HuggingFace parameter naming** — `model.layers.0.self_attn.q_proj.weight` | What names HF models expect in `state_dict` |
| **`str.format(i=i)`** — named string formatting | Template expansion for layer indices |
| **`model.load_state_dict(d, strict=False)`** — loads weights into a model | How you inject the dequantized tensors |

### `src/gguf/loader.py`

### Concepts it teaches:
- **The GGUF-to-HuggingFace name map** — `blk.0.attn_q.weight` → `model.layers.0.self_attn.q_proj.weight`. Two naming conventions for the same weights.
- **Template expansion** — `GGUF_TO_HF` stores patterns with `{i}`. `_build_name_map(n_layers)` expands them for all layer indices.
- **Auto-detection from metadata** — `n_layers = reader.metadata.get("llama.block_count", 32)` reads the layer count from the file instead of hardcoding it.
- **Two-pass loading** — first collect all names to load, then load each (avoids modifying a dict while iterating).

---

### Before reading `tests/test_gguf_reader.py`:

| Concept | Why you need it |
|---------|----------------|
| `k_quants.py` and `reader.py` (above) | What we're testing |
| **Synthetic test data** — constructing known bytes with `struct.pack` | Instead of downloading a real GGUF file |
| **`pytest.mark.skipif`** — conditionally skip a test | The real-file test is skipped if no `.gguf` exists |

### `tests/test_gguf_reader.py`

### Concepts it teaches:
- **Testing binary parsers with synthetic bytes** — construct a block with `struct.pack("<e", 1.0) + struct.pack("<32b", *values)` and verify the dequantizer gives the expected output. No real model file needed.
- **Known-value testing** — `test_q8_0_simple` uses `scale=1.0` and `values=[1..32]` so the expected output is trivially `[1.0..32.0]`. This is cleaner than comparing to a reference implementation.
- **Multiple blocks** — `test_q8_0_multiple_blocks` concatenates two blocks and verifies the output is the correct concatenation. Tests the block-stride arithmetic.
- **Skippable integration tests** — the real GGUF test uses `pytest.mark.skipif` with a helper that searches for `.gguf` files. It teaches how to write tests that work in CI (without model files) and also work locally (with real files).

---

## Part 6 — Applying Everything to a Real Model

---

### Before reading `scripts/quantize_model.py`:

| Concept | Why you need it |
|---------|----------------|
| All of `src/quantization/` | The script calls all quantizers |
| **`argparse`** — `parser.add_argument("--scheme", ...)` | How CLI arguments are parsed |
| **`model.named_modules()`** — iterating all layers | How the script finds every `nn.Linear` |
| **`copy.deepcopy(model)`** — cloning a model | So we can compare before/after |
| `datasets.load_dataset("wikitext", ...)` | Loading calibration data for GPTQ |

### `scripts/quantize_model.py`

### Concepts it teaches:
- **End-to-end quantization workflow** — load model → apply scheme → measure size → generate text → save.
- **Module replacement** — to replace `nn.Linear` with `QuantizedLinear`, navigate to the parent module with `getattr(parent, part)` and call `setattr(parent, last_part, new_module)`.
- **Two-pass replacement** — collect `{name: new_module}` in a dict first, then apply. Replacing while iterating `named_modules()` causes undefined behavior.
- **Print-as-you-go progress** — for GPTQ (which takes minutes), printing `[1/22] blk.0.attn_q...` gives visual feedback on a slow operation.

---

### Before reading `scripts/eval_perplexity.py`:

| Concept | Why you need it |
|---------|----------------|
| `src/eval/perplexity.py` | The script is a CLI wrapper around it |
| `scripts/quantize_model.py` patterns | Same quantization application logic |

### `scripts/eval_perplexity.py`

### Concepts it teaches:
- **CLI flag design** — `--quick` (fast estimate on validation set) vs full test set. Always give users a fast path for development.
- **Dtype selection** — `torch.float32` for fp32 mode, `torch.float16` for everything else. The model loads at the right precision from the start.

---

## Part 7 — Benchmarking All Schemes Systematically

---

### Before reading `benchmarks/benchmark_quant.py`:

| Concept | Why you need it |
|---------|----------------|
| All modules above | The benchmark calls everything |
| **JSON serialization** — `json.dump(results, f)` | How results are saved and loaded |
| **Intermediate saves** — saving after each scheme | So a long benchmark isn't lost if it crashes |
| **`vars(dataclass_instance)`** — converts dataclass to dict | Needed for JSON serialization |

### `benchmarks/benchmark_quant.py`

### Concepts it teaches:
- **Systematic benchmarking** — run all schemes in the same conditions (same model, same hardware, same test data) so comparisons are fair.
- **One fresh model load per scheme** — calling `load_fresh_model()` each time ensures no scheme benefits from (or is hurt by) residual state from a previous scheme.
- **Intermediate result saving** — saves `quant_results.json` after every scheme. If GPTQ crashes after 3 hours, you still have the first 3 results.
- **Graceful failure** — wrap each scheme in `try/except` so a failing scheme doesn't kill the whole benchmark run.

---

### Before reading `benchmarks/benchmark_gguf.py`:

| Concept | Why you need it |
|---------|----------------|
| `src/gguf/reader.py` | The benchmark uses GGUFReader directly |
| `os.path.getsize()` | File size measurement |
| `time.time()` around operations | Timing the parse and load steps |

### `benchmarks/benchmark_gguf.py`

### Concepts it teaches:
- **Lazy parse vs full load** — parsing metadata takes milliseconds; loading all tensors takes seconds. The benchmark measures both separately.
- **Cross-format comparison** — loading a Q4_K_M tensor from GGUF and then applying our own absmax INT8 on top shows how the formats compare numerically.
- **File size vs memory footprint** — a 750 MB GGUF file becomes ~4 GB of float32 tensors when fully dequantized. The benchmark prints both.

---

## Part 8 — The API Server

---

### Before reading `src/api/server.py`:

| Concept | Why you need it |
|---------|----------------|
| **FastAPI basics** — `@app.post("/route")`, `@app.get("/route")` | How REST endpoints are defined |
| **Pydantic models** — `class MyModel(BaseModel): field: type` | FastAPI uses these for request/response validation |
| **`uvicorn`** — the ASGI server that runs FastAPI | What actually serves HTTP requests |
| All quantization modules | The server calls them inside the endpoint |

### `src/api/server.py`

### Concepts it teaches:
- **Pydantic for validation** — `QuantizeRequest` validates that `scheme` is one of the allowed strings, `calibration_samples` is an int, etc. Bad input is rejected with a clear error before your code even runs.
- **`HTTPException`** — raising this inside a FastAPI endpoint sends an HTTP error response (e.g. 404 or 400) to the caller.
- **`response_model`** — `@app.post("/quantize", response_model=QuantizeResponse)` tells FastAPI exactly what shape to return and validates it automatically.
- **Device detection at request time** — detect MPS/CUDA/CPU inside the endpoint rather than at startup, so the server works on any hardware without reconfiguration.

### `scripts/run_server.py`

### Concepts it teaches:
- **`uvicorn.run("module:app", ...)`** — the string form imports the app from the module. This is the standard way to run FastAPI in development.
- **`--reload` flag** — restarts the server when any Python file changes. Useful during development, never use in production.

---

## Part 9 — Notebooks (Pull Everything Together Visually)

Follow notebooks in order. Each one assumes you've read the corresponding module above.

---

### Before opening `notebooks/01_absmax_zeropoint.ipynb`:

| Concept | Why you need it |
|---------|----------------|
| `src/quantization/absmax.py` and `zeropoint.py` | The notebook calls these directly |
| **`matplotlib.pyplot`** — `plt.hist()`, `plt.bar()`, `plt.imshow()` | All charts use matplotlib |
| **`%matplotlib inline`** — Jupyter magic command | Displays charts inside the notebook |

### `notebooks/01_absmax_zeropoint.ipynb`

### Concepts it teaches — by seeing them:
- The actual integer values after quantization (bar chart)
- Quantization error visualized as a bar chart per weight
- INT8 vs INT4 error side by side — you see INT4 is much noisier
- Real TinyLlama weight histogram — nearly Gaussian, symmetric around zero
- Per-tensor vs per-channel error heatmaps — per-channel is visibly darker (less error)
- Before/after generation comparison — you see that INT8 barely changes the output

---

### Before opening `notebooks/02_gptq_walkthrough.ipynb`:

| Concept | Why you need it |
|---------|----------------|
| `src/quantization/gptq.py` and `gptq_utils.py` | The notebook runs them |
| Notebook 01 (above) | Builds on the error comparison visualizations |
| **`plt.imshow(matrix)`** | Heatmap visualization |

### `notebooks/02_gptq_walkthrough.ipynb`

### Concepts it teaches — by seeing them:
- The Hessian matrix as an image — bright diagonal = important weights
- `H @ H_inv ≈ I` visualized — confirms the inversion is correct
- GPTQ vs naive INT4 error heatmaps side by side — GPTQ is visibly darker
- The INT4 packing arithmetic printed step by step — you see the exact bytes

---

### Before opening `notebooks/03_gguf_inspection.ipynb`:

| Concept | Why you need it |
|---------|----------------|
| `src/gguf/reader.py` and `k_quants.py` | The notebook uses GGUFReader |
| A downloaded `.gguf` file | Download with `huggingface-cli download` as shown in notebook |
| `docs/gguf_format.md` | Explains what you're about to see in the file |

### `notebooks/03_gguf_inspection.ipynb`

### Concepts it teaches — by seeing them:
- All metadata key-value pairs printed — you see the full model config in the file
- Tensor names and shapes — see `blk.0.attn_q.weight`, `blk.1.ffn_gate.weight`, etc.
- Raw bytes shown next to dequantized values for Q8_0 — the connection between disk and memory
- File size (MB) vs FP32 equivalent printed — the compression ratio made concrete

---

### Before opening `notebooks/04_benchmark_dashboard.ipynb`:

| Concept | Why you need it |
|---------|----------------|
| All previous notebooks | This is the capstone visualization |
| Benchmark results JSON | Run `benchmark_quant.py --quick` first |
| **`plotly` or `matplotlib` scatter plots** | Charts 1 and 4 use scatter |

### `notebooks/04_benchmark_dashboard.ipynb`

### Concepts it teaches — by seeing them:
- **Pareto frontier** — which schemes are on the optimal memory/quality curve
- **The 4-bit quality cliff** — chart 4 shows perplexity rising sharply below 4 bits
- **Error heatmap comparison** — INT8 vs naive INT4 vs GPTQ INT4, all three side by side
- **Speed vs size** — smaller models are faster because weights load from memory faster

---

## Reading Order Summary

```
FULL PROJECT CONCEPTS (Part 0)
        │
        ▼
absmax.py → zeropoint.py → dequantize.py
        │
        ▼
tests/test_absmax.py → tests/test_zeropoint.py
        │
        ▼
eval/metrics.py → eval/perplexity.py
        │
        ▼
gptq_utils.py → gptq.py → tests/test_gptq.py
        │
        ▼
gguf/k_quants.py → gguf/reader.py → gguf/loader.py → tests/test_gguf_reader.py
        │
        ▼
scripts/quantize_model.py → scripts/eval_perplexity.py
        │
        ▼
benchmarks/benchmark_quant.py → benchmarks/benchmark_gguf.py
        │
        ▼
api/server.py → scripts/run_server.py
        │
        ▼
notebooks/01 → 02 → 03 → 04
        │
        ▼
docs/results_analysis.md  (fill in your actual numbers here)
```

**One rule:** If a concept in a file feels unexplained, scroll up in this document to the section for that file's prerequisites. The answer is there.
