[![Discord](https://img.shields.io/discord/232596713892872193?logo=discord)](https://discord.gg/2JhHVh7CGu)

Gated attention transformer using Deep Seek's routing with a general large MoE.

Architecture:
- Deep seek style MoE (Auxillary loss free routing: https://arxiv.org/abs/2408.15664)
- Meituan Long Cat partial magnitude-corrected routing (For early layer routing imbalance) (LongCat-Flash technical report https://arxiv.org/html/2509.01322v1)
- Zero Centered RMS Norm /w Weight Decay (Concept from Qwen3-Next: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- Gated Attention (https://arxiv.org/abs/2505.06708)
- Scattermoe (https://github.com/shawntan/scattermoe/tree/main)
- Direction decoupled muon optimizer (https://haeggee.github.io/posts/magnitude-direction-decoupling)
- Cut cross entropy training (https://arxiv.org/abs/2411.09009)

### Do the thing
Using uv:
```
uv sync
```

Train against pre-tokenized JSONL data under `outputs/` by default:
```
uv run python main.py
```

The training run uses ordinary next-token cut cross entropy with partial
positional RoPE. Documents are packed into a single flat, unpadded token stream
(FlashAttention varlen via `cu_seqlens`), so the model runs on real tokens only —
no padding is carried through the layer stack.

Aim logs to: `logs/aim`
```
uv run aim up --repo logs/aim
```
