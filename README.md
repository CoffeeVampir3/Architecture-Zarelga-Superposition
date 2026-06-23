[![Discord](https://img.shields.io/discord/232596713892872193?logo=discord)](https://discord.gg/2JhHVh7CGu)

Gated attention transformer using Deep Seek's routing with a general large MoE.

Architecture:
- Deep seek style MoE (Auxillary loss free routing: https://arxiv.org/abs/2408.15664)
- Meituan Long Cat partial magnitude-corrected routing (For early layer routing imbalance) (LongCat-Flash technical report https://arxiv.org/html/2509.01322v1)
- Zero Centered RMS Norm /w Weight Decay (Concept from Qwen3-Next: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- Gated Attention (https://arxiv.org/abs/2505.06708)
- Token superposition/Patch training (https://arxiv.org/pdf/2605.06546 / https://arxiv.org/abs/2407.12665)
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

The default training run uses ordinary next-token cut cross entropy with
partial positional RoPE. Token Superposition Training, S-RoPE, and the
MCCE-backed multi-target loss are disabled on the normal `main.py` path; flip
`set_superposition(..., enabled=True)` when running the A/B variant.

Aim logs to: `logs/aim`
```
uv run aim up --repo logs/aim
```
