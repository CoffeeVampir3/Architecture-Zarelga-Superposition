[![Discord](https://img.shields.io/discord/232596713892872193?logo=discord)](https://discord.gg/2JhHVh7CGu)

Gated attention transformer using Deep Seek's routing with a general large MoE.

Architecture:
- Deep seek style MoE (Auxillary loss free routing: https://arxiv.org/abs/2408.15664)
- Zero Centered RMS Norm /w Weight Decay (Concept from Qwen3-Next: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- Gated Attention (https://arxiv.org/abs/2505.06708)

Auxillary stuff:
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

Aim logs to: `logs/aim`
```
uv run aim up --repo logs/aim
```
