"""
Module 3C: GGUF Model Loader
==============================

Bridges the gap between GGUF tensor names and HuggingFace model parameter names.

GGUF uses names like:   blk.0.attn_q.weight
HuggingFace uses:       model.layers.0.self_attn.q_proj.weight

This loader reads a GGUF file, dequantizes all tensors, and returns a
state_dict that can be loaded directly into a HuggingFace LLaMA model.

Why would you do this instead of just using llama.cpp?
- To understand exactly how the format works
- To load a GGUF model into PyTorch for fine-tuning or analysis
- To mix GGUF-format weights with PyTorch code
"""

import torch
from typing import Optional
from .reader import GGUFReader


# Mapping from GGUF tensor names to HuggingFace LLaMA names
# {i} is replaced with the actual layer index
GGUF_TO_HF = {
    # Embeddings
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",

    # Transformer layer patterns (use .format(i=i))
    "blk.{i}.attn_norm.weight":   "model.layers.{i}.input_layernorm.weight",
    "blk.{i}.ffn_norm.weight":    "model.layers.{i}.post_attention_layernorm.weight",
    "blk.{i}.attn_q.weight":      "model.layers.{i}.self_attn.q_proj.weight",
    "blk.{i}.attn_k.weight":      "model.layers.{i}.self_attn.k_proj.weight",
    "blk.{i}.attn_v.weight":      "model.layers.{i}.self_attn.v_proj.weight",
    "blk.{i}.attn_output.weight": "model.layers.{i}.self_attn.o_proj.weight",
    "blk.{i}.ffn_gate.weight":    "model.layers.{i}.mlp.gate_proj.weight",
    "blk.{i}.ffn_up.weight":      "model.layers.{i}.mlp.up_proj.weight",
    "blk.{i}.ffn_down.weight":    "model.layers.{i}.mlp.down_proj.weight",
}


class GGUFModelLoader:
    """
    Load a GGUF file and return a HuggingFace-compatible state dict.

    Usage:
        loader = GGUFModelLoader()
        state_dict = loader.load("model.gguf", n_layers=32)
        hf_model.load_state_dict(state_dict, strict=False)
    """

    def __init__(self):
        self.reader = None

    def load(self, gguf_path: str, n_layers: Optional[int] = None) -> dict[str, torch.Tensor]:
        """
        Read all tensors from a GGUF file and convert names to HuggingFace format.

        Args:
            gguf_path: Path to the .gguf file
            n_layers:  Number of transformer layers (auto-detected if None)

        Returns:
            state_dict: {hf_param_name: float32_tensor}
        """
        self.reader = GGUFReader(gguf_path)

        # Auto-detect number of layers from metadata if not provided
        if n_layers is None:
            n_layers = self.reader.metadata.get("llama.block_count", 32)
            print(f"Auto-detected n_layers = {n_layers}")

        # Build a lookup: gguf_name → hf_name for all layers
        name_map = self._build_name_map(n_layers)

        state_dict = {}
        tensor_names = {t.name for t in self.reader.tensor_infos}

        loaded = 0
        skipped = 0

        for gguf_name, hf_name in name_map.items():
            if gguf_name not in tensor_names:
                skipped += 1
                continue

            print(f"  Loading {gguf_name:50s} → {hf_name}")
            try:
                tensor = self.reader.load_tensor(gguf_name)
                state_dict[hf_name] = tensor
                loaded += 1
            except Exception as e:
                print(f"  Warning: Failed to load '{gguf_name}': {e}")
                skipped += 1

        print(f"\nLoaded {loaded} tensors, skipped {skipped}")
        return state_dict

    def _build_name_map(self, n_layers: int) -> dict[str, str]:
        """Expand the template name map for all layer indices."""
        name_map = {}

        for gguf_template, hf_template in GGUF_TO_HF.items():
            if "{i}" in gguf_template:
                # Expand for each layer
                for i in range(n_layers):
                    gguf_name = gguf_template.format(i=i)
                    hf_name = hf_template.format(i=i)
                    name_map[gguf_name] = hf_name
            else:
                name_map[gguf_template] = hf_template

        return name_map

    def get_model_info(self, gguf_path: str) -> dict:
        """
        Quickly read GGUF metadata without loading tensor data.
        Useful for inspecting a file before committing to loading it.
        """
        reader = GGUFReader(gguf_path)
        info = {
            "path": gguf_path,
            "n_tensors": len(reader.tensor_infos),
            "metadata": dict(reader.metadata),
            "tensors": [
                {
                    "name": t.name,
                    "shape": t.shape,
                    "dtype": t.dtype_name,
                    "size_mb": t.size_bytes / (1024 ** 2),
                }
                for t in reader.tensor_infos
            ],
        }
        return info
