#!/usr/bin/env python3
"""Add per-layer RoPE and sliding-window support to vLLM's Qwen3 model.

Kanana 2 1.3B alternates full-attention layers using YaRN RoPE with
sliding-attention layers using the default RoPE and a 1,024-token window. Its
``rope_parameters`` value is keyed by layer type, while vLLM 0.19 passes the
nested mapping directly to ``get_rope`` and raises ``TypeError``. The patch is
idempotent and preserves the original module as ``qwen3.py.orig``.
"""
import shutil, sys
from pathlib import Path

import vllm
Q = Path(vllm.__file__).parent / "model_executor" / "models" / "qwen3.py"
src = Q.read_text()
legacy_marker = "D" + "APR_PATCH"
if "PRECALL_GATING_PATCH" in src or legacy_marker in src:
    print("이미 패치됨:", Q); sys.exit(0)
shutil.copy(Q, str(Q) + ".orig")

# Extend Qwen3Attention with a per-layer sliding-window argument.
old_sig = """        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size"""
new_sig = """        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
        per_layer_sliding_window: int | None = None,  # PRECALL_GATING_PATCH
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size"""
assert old_sig in src, "signature anchor not found"
src = src.replace(old_sig, new_sig, 1)

# Forward the per-layer sliding-window value to Attention.
old_attn = """            attn_type=attn_type,"""
assert src.count(old_attn) >= 1
src = src.replace(old_attn,
                  """            attn_type=attn_type,
            per_layer_sliding_window=per_layer_sliding_window,  # PRECALL_GATING_PATCH""", 1)

# Select RoPE parameters and window size for each decoder layer type.
old_dec = """        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,"""
new_dec = """        # PRECALL_GATING_PATCH: per-layer RoPE and sliding window
        _rope_parameters = config.rope_parameters
        _sliding_window = None
        _layer_types = getattr(config, "layer_types", None)
        if _layer_types:
            from vllm.model_executor.models.utils import extract_layer_index
            _lt = _layer_types[extract_layer_index(prefix)]
            if _lt == "sliding_attention" and getattr(config, "sliding_window", None):
                _sliding_window = config.sliding_window
            if isinstance(_rope_parameters, dict) and _lt in _rope_parameters:
                _rope_parameters = _rope_parameters[_lt]
        # PRECALL_GATING_PATCH end

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,"""
assert old_dec in src, "decoder anchor not found"
src = src.replace(old_dec, new_dec, 1)

old_pass = """            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )"""
new_pass = """            rope_parameters=_rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            per_layer_sliding_window=_sliding_window,  # PRECALL_GATING_PATCH
        )"""
assert old_pass in src, "pass anchor not found"
src = src.replace(old_pass, new_pass, 1)

Q.write_text(src)
print("패치 완료:", Q)
print("원본 백업:", str(Q) + ".orig")
