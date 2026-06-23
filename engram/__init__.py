"""Shippable Engram sparse path — the two standalone components, torch-only.

This package is the deployable subset of ``engram.prod``: the sparse-gradient
embedding and its matching per-row optimizer. Both modules are self-contained
(``torch`` only, no intra-package imports), so this folder can be copied as-is into
another project and imported directly.

    from shipping import EngramConfig, EngramEmbeddingSparse, GramReaperSparse

    cfg = EngramConfig(vocab_size=50257, d_model=768, layer_id=6,
                       importance_weighting=True)
    engram = EngramEmbeddingSparse(cfg)
    # The sparsely-addressed tables train on the per-row optimizer; the dense
    # side-modules (value_proj, alpha) ride a normal optimizer (e.g. Adam).
    table_params = [engram.embedding.weight]
    if engram.importance_weighting:
        table_params.append(engram.imp_table.weight)
    table_opt = GramReaperSparse(table_params, lr=1e-2)
    side_opt = torch.optim.Adam(
        [engram.value_proj.weight, engram.value_proj.bias, engram.alpha], lr=1e-2)

    hidden = engram(hidden_states, token_ids)   # [B, T, d_model], [B, T]
"""

from .EngramEmbeddingSparse import EngramConfig, EngramEmbeddingSparse
from .GramReaperSparse import GramReaperSparse

__all__ = [
    "EngramConfig",
    "EngramEmbeddingSparse",
    "GramReaperSparse",
]
