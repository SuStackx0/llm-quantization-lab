"""
Module 3A: GGUF File Reader
============================

GGUF (GGML Universal Format) is the binary file format used by:
- llama.cpp  (the most popular C++ inference engine)
- Ollama     (Docker-style LLM serving)
- LM Studio  (local LLM GUI)

Understanding GGUF means you understand how these tools store models on disk.

File layout:
┌──────────────────────────────────────────┐
│  HEADER                                  │
│  - 4 bytes: magic ("GGUF")               │
│  - 4 bytes: version (2 or 3)             │
│  - 8 bytes: n_tensors (count)            │
│  - 8 bytes: n_kv (metadata count)        │
├──────────────────────────────────────────┤
│  METADATA (n_kv entries)                 │
│  - string key, uint32 type, value        │
│  (model name, context length, vocab, ...) │
├──────────────────────────────────────────┤
│  TENSOR INFO (n_tensors entries)          │
│  - string name, shape, dtype, byte offset│
├──────────────────────────────────────────┤
│  ALIGNMENT PADDING                       │
├──────────────────────────────────────────┤
│  TENSOR DATA (raw bytes, back-to-back)   │
└──────────────────────────────────────────┘
"""

import struct
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, BinaryIO
from .k_quants import dequantize_q4_k, dequantize_q8_0, dequantize_q5_k


# GGUF quantization type IDs → human-readable names
GGUF_TYPE_NAMES = {
    0:  "F32",
    1:  "F16",
    2:  "Q4_0",
    3:  "Q4_1",
    6:  "Q5_0",
    7:  "Q5_1",
    8:  "Q8_0",
    9:  "Q8_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q4_K_S",
    13: "Q4_K_M",
    14: "Q5_K_S",
    15: "Q5_K_M",
    16: "Q6_K",
    17: "Q8_K",
}

# GGUF metadata value types
GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}


@dataclass
class TensorInfo:
    """Metadata about one tensor in the GGUF file (not the data itself)."""
    name: str
    shape: tuple
    dtype_id: int
    dtype_name: str
    byte_offset: int      # where the raw tensor data starts in the file
    n_elements: int       # total number of elements

    @property
    def size_bytes(self) -> int:
        """Approximate file size of this tensor in bytes."""
        bits_per_element = {
            0: 32, 1: 16, 2: 4, 3: 4, 6: 5, 7: 5,
            8: 8, 9: 8, 12: 4, 13: 4, 14: 5, 15: 5, 16: 6, 17: 8
        }
        bpe = bits_per_element.get(self.dtype_id, 32)
        return (self.n_elements * bpe) // 8


class GGUFReader:
    """
    Parse a GGUF file without loading all tensor data into memory at once.

    Usage:
        reader = GGUFReader("llama-3.2-1b-q4_k_m.gguf")
        print(reader.metadata["general.name"])
        for t in reader.tensor_infos:
            print(t.name, t.shape, t.dtype_name)
        tensor = reader.load_tensor("blk.0.attn_q.weight")
    """

    MAGIC = b"GGUF"
    SUPPORTED_VERSIONS = [2, 3]

    def __init__(self, path: str):
        self.path = path
        self.metadata = {}
        self.tensor_infos = []
        self._data_offset = 0  # byte position where tensor data begins

        with open(path, "rb") as f:
            self._parse_header(f)
            self._parse_metadata(f)
            self._parse_tensor_infos(f)
            # Tensor data starts at the next 32-byte aligned position
            current_pos = f.tell()
            alignment = 32
            self._data_offset = ((current_pos + alignment - 1) // alignment) * alignment

        print(f"Loaded GGUF: {len(self.tensor_infos)} tensors, {len(self.metadata)} metadata entries")

    def _parse_header(self, f: BinaryIO):
        magic = f.read(4)
        if magic != self.MAGIC:
            raise ValueError(f"Not a GGUF file — expected magic 'GGUF', got {magic}")

        version = self._read_uint32(f)
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported GGUF version {version}. Supported: {self.SUPPORTED_VERSIONS}")
        self.version = version

        self._n_tensors = self._read_uint64(f)
        self._n_kv = self._read_uint64(f)

    def _parse_metadata(self, f: BinaryIO):
        """Read all key-value metadata entries."""
        for _ in range(self._n_kv):
            key = self._read_string(f)
            value_type_id = self._read_uint32(f)
            value = self._read_value(f, value_type_id)
            self.metadata[key] = value

    def _parse_tensor_infos(self, f: BinaryIO):
        """Read info about each tensor (name, shape, dtype, offset)."""
        for _ in range(self._n_tensors):
            name = self._read_string(f)
            n_dims = self._read_uint32(f)

            # Dimensions are stored in GGUF as [fastest-varying, ..., slowest-varying]
            # which is the reverse of PyTorch/NumPy convention
            dims = [self._read_uint64(f) for _ in range(n_dims)]
            shape = tuple(reversed(dims))  # convert to standard [out, in] order

            dtype_id = self._read_uint32(f)
            dtype_name = GGUF_TYPE_NAMES.get(dtype_id, f"unknown_{dtype_id}")
            byte_offset = self._read_uint64(f)

            n_elements = 1
            for d in shape:
                n_elements *= d

            self.tensor_infos.append(TensorInfo(
                name=name,
                shape=shape,
                dtype_id=dtype_id,
                dtype_name=dtype_name,
                byte_offset=byte_offset,
                n_elements=n_elements,
            ))

    def load_tensor(self, name: str) -> torch.Tensor:
        """
        Load and dequantize a tensor by name.

        The raw bytes are read from disk, then decoded according to the
        quantization format (Q4_K, Q8_0, F16, etc.).

        Returns float32 tensor.
        """
        # Find tensor metadata
        info = None
        for t in self.tensor_infos:
            if t.name == name:
                info = t
                break
        if info is None:
            available = [t.name for t in self.tensor_infos[:10]]
            raise KeyError(f"Tensor '{name}' not found. First 10 names: {available}")

        # Read raw bytes from disk
        with open(self.path, "rb") as f:
            abs_offset = self._data_offset + info.byte_offset
            f.seek(abs_offset)
            raw_bytes = f.read(info.size_bytes)

        # Dispatch to the appropriate dequantization function
        return self._dequantize(raw_bytes, info)

    def _dequantize(self, data: bytes, info: TensorInfo) -> torch.Tensor:
        """Choose the right dequantization function based on dtype."""
        dtype_id = info.dtype_id

        if dtype_id == 0:   # F32
            arr = np.frombuffer(data, dtype=np.float32)
            return torch.from_numpy(arr.copy()).reshape(info.shape)

        elif dtype_id == 1:  # F16
            arr = np.frombuffer(data, dtype=np.float16)
            return torch.from_numpy(arr.copy()).float().reshape(info.shape)

        elif dtype_id == 8:  # Q8_0
            return dequantize_q8_0(data, info.shape)

        elif dtype_id in (12, 13):  # Q4_K_S or Q4_K_M
            return dequantize_q4_k(data, info.shape)

        elif dtype_id in (14, 15):  # Q5_K_S or Q5_K_M
            return dequantize_q5_k(data, info.shape)

        else:
            raise NotImplementedError(
                f"Dequantization for {info.dtype_name} (id={dtype_id}) not yet implemented. "
                f"Supported: F32, F16, Q8_0, Q4_K, Q5_K"
            )

    def summary(self) -> str:
        """Pretty-print a summary of the GGUF file contents."""
        lines = [f"GGUF File: {self.path}", f"Version: {self.version}"]

        if "general.name" in self.metadata:
            lines.append(f"Model: {self.metadata['general.name']}")
        if "general.architecture" in self.metadata:
            lines.append(f"Architecture: {self.metadata['general.architecture']}")

        lines.append(f"\nTensors ({len(self.tensor_infos)}):")
        for t in self.tensor_infos[:20]:  # show first 20
            size_mb = t.size_bytes / (1024 ** 2)
            lines.append(f"  {t.name:50s}  {str(t.shape):25s}  {t.dtype_name:10s}  {size_mb:.2f} MB")
        if len(self.tensor_infos) > 20:
            lines.append(f"  ... and {len(self.tensor_infos) - 20} more")

        total_mb = sum(t.size_bytes for t in self.tensor_infos) / (1024 ** 2)
        lines.append(f"\nTotal tensor data: {total_mb:.1f} MB")
        return "\n".join(lines)

    # --- Binary reading helpers ---

    def _read_uint8(self, f):  return struct.unpack("<B", f.read(1))[0]
    def _read_int8(self, f):   return struct.unpack("<b", f.read(1))[0]
    def _read_uint16(self, f): return struct.unpack("<H", f.read(2))[0]
    def _read_int16(self, f):  return struct.unpack("<h", f.read(2))[0]
    def _read_uint32(self, f): return struct.unpack("<I", f.read(4))[0]
    def _read_int32(self, f):  return struct.unpack("<i", f.read(4))[0]
    def _read_uint64(self, f): return struct.unpack("<Q", f.read(8))[0]
    def _read_int64(self, f):  return struct.unpack("<q", f.read(8))[0]
    def _read_float32(self, f): return struct.unpack("<f", f.read(4))[0]
    def _read_float64(self, f): return struct.unpack("<d", f.read(8))[0]
    def _read_bool(self, f):   return struct.unpack("<?", f.read(1))[0]

    def _read_string(self, f) -> str:
        length = self._read_uint64(f)
        return f.read(length).decode("utf-8", errors="replace")

    def _read_value(self, f, type_id: int):
        """Read a metadata value of the given type."""
        readers = {
            0: self._read_uint8,
            1: self._read_int8,
            2: self._read_uint16,
            3: self._read_int16,
            4: self._read_uint32,
            5: self._read_int32,
            6: self._read_float32,
            7: self._read_bool,
            8: self._read_string,
            10: self._read_uint64,
            11: self._read_int64,
            12: self._read_float64,
        }
        if type_id == 9:  # array
            elem_type = self._read_uint32(f)
            n = self._read_uint64(f)
            # Read first few elements and truncate for large arrays
            items = []
            limit = min(n, 100)
            for _ in range(n):
                if _ < limit and elem_type in readers:
                    items.append(readers[elem_type](f))
                elif elem_type == 8:
                    _ = self._read_string(f)  # consume but don't store
                elif elem_type in readers:
                    readers[elem_type](f)
            return items

        reader = readers.get(type_id)
        if reader is None:
            return None
        return reader(f)
