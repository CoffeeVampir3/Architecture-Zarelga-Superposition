"""Canonical token-ID map for Engram addressing (tokenizer compression).

Textually-equivalent tokens (case, accents, space-run variants) are merged to one
canonical ID before hashing, pooling their n-gram statistics into shared rows.
Only Engram addressing sees canonical IDs; the backbone embedding does not.

Two deliberate deviations from a naive normalizer (both verified on the repo
tokenizer; see runs/engram_ablating/EXPERIMENT_REPORT.md, phase 5):
  * byte-fallback / undecodable tokens (decode contains U+FFFD) stay distinct —
    they are different bytes, not textual variants;
  * only space runs collapse; newlines are preserved (paragraph structure matters).
"""

from __future__ import annotations

import re
import unicodedata

import torch


def _canon(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    return re.sub(r" +", " ", s)


def build_token_canon(tokenizer) -> torch.Tensor:
    """``[vocab]`` int64 map: token ID -> canonical token ID (surjective, id-stable:
    each group maps to its lowest member ID). Specials map to themselves."""
    vocab = len(tokenizer)
    specials = {tokenizer.pad_token_id, getattr(tokenizer, "eos_token_id", None)} - {None}
    groups: dict[str, int] = {}
    out = torch.arange(vocab, dtype=torch.int64)
    for i in range(vocab):
        s = tokenizer.decode([i])
        if i in specials or not s or "�" in s:
            continue
        out[i] = groups.setdefault(_canon(s), i)
    return out


def apply_token_canon(model, tokenizer) -> int:
    """Fill every Engram module's ``token_canon`` buffer; returns merged-ID count."""
    engrams = getattr(model, "engrams", None)
    if engrams is None:
        return 0
    canon = build_token_canon(tokenizer)
    merged = int(canon.numel() - canon.unique().numel())
    for engram in engrams.values():
        canon_dev = canon.to(engram.token_canon.device)
        engram.token_canon.copy_(canon_dev)
    return merged
