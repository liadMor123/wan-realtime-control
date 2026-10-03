"""Locate umT5 sentencepiece positions of the timed object and the temporal phrase.

Positions index the 512-slot context exactly as WanModel sees it: the T5 output
is trimmed to the real tokens (including </s>) and zero-padded to 512, and the
padding slots are attended (WanModel passes context_lens=None to cross-attn).
"""
import importlib.util
import os
import re

from .masks import TEMPORAL_PHRASE_RE

_WAN_TOKENIZERS = None


def _wan_tokenizers_module(wan_root):
    """Import wan/modules/tokenizers.py without wan/__init__ (which needs CUDA)."""
    global _WAN_TOKENIZERS
    if _WAN_TOKENIZERS is None:
        path = os.path.join(wan_root, "wan", "modules", "tokenizers.py")
        spec = importlib.util.spec_from_file_location("wan_tokenizers", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _WAN_TOKENIZERS = mod
    return _WAN_TOKENIZERS


def load_tokenizer(wan_root, ckpt_dir, text_len=512):
    mod = _wan_tokenizers_module(wan_root)
    return mod.HuggingfaceTokenizer(
        name=os.path.join(ckpt_dir, "google", "umt5-xxl"), seq_len=text_len, clean="whitespace")


def span_token_indices(tok, prompt, spans):
    """Token indices overlapping any character span, on the cleaned prompt Wan encodes."""
    cleaned = tok._clean(prompt)
    if cleaned != prompt:
        raise ValueError(f"Wan's whitespace clean changes the prompt; offsets would drift: {prompt!r}")
    enc = tok.tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    ids_wan = tok([prompt], return_mask=False)[0].tolist()
    n_real = len(enc["input_ids"])
    if ids_wan[:n_real] != enc["input_ids"]:
        raise ValueError("offset tokenization disagrees with Wan's tokenizer ids")
    idx = set()
    for a, b in spans:
        for i, (s, e) in enumerate(enc["offset_mapping"]):
            if e > s and not (e <= a or s >= b):
                idx.add(i)
    return sorted(idx), n_real, enc


def object_token_indices(tok, prompt, obj):
    """All occurrences of `obj` as a whole word (TempoControl uses raw substring find;
    whole-word matching is identical on the benchmark and avoids e.g. 'cat' in 'catch')."""
    spans = [m.span() for m in re.finditer(r"\b" + re.escape(obj) + r"\b", prompt)]
    if not spans:
        raise ValueError(f"object {obj!r} not found in prompt {prompt!r}")
    idx, n_real, enc = span_token_indices(tok, prompt, spans)
    if not idx:
        raise ValueError(f"no tokens for object {obj!r}")
    return idx, n_real, [tok.tokenizer.convert_ids_to_tokens(enc["input_ids"][i]) for i in idx]


def temporal_token_indices(tok, prompt):
    spans = [m.span() for m in TEMPORAL_PHRASE_RE.finditer(prompt)]
    if not spans:
        raise ValueError(f"no temporal phrase in prompt {prompt!r}")
    idx, n_real, enc = span_token_indices(tok, prompt, spans)
    return idx, n_real, [tok.tokenizer.convert_ids_to_tokens(enc["input_ids"][i]) for i in idx]
