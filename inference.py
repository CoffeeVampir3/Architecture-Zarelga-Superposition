import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

from modeling.model import MoEModel
from current_configuration import build_config


ROOT = Path(__file__).resolve().parent
DEFAULT_TOKENIZER = ROOT / "tokenizer.json"
EOS_TOKEN = "<|endoftext|>"


def load_tokenizer(tokenizer_path: Path):
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        eos_token=EOS_TOKEN,
        pad_token=EOS_TOKEN,
    )


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, top_k: int = 5,
             temperature: float = 0.0, device: str = "cuda"):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)

    sampling = temperature > 0.0
    mode = f"sampling T={temperature:g}" if sampling else "greedy (argmax)"

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    for step in range(max_new_tokens):
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(tokens)

        next_logits = logits[:, -1].float()
        probs = torch.softmax(next_logits, dim=-1)

        if sampling:
            scaled = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(scaled, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        chosen_id = next_token.item()
        chosen_prob = probs[0, chosen_id].item()

        topk_probs, topk_ids = probs[0].topk(top_k)
        topk_id_list = topk_ids.tolist()
        token_text = tokenizer.decode([chosen_id], skip_special_tokens=False)
        print(
            f"\n[step {step:>3}] {mode} -> {chosen_id} "
            f"({token_text!r}) p={chosen_prob:.4f}  "
            f"entropy={_entropy(probs[0]):.3f} nats"
        )
        for rank, (p, tid) in enumerate(zip(topk_probs.tolist(), topk_id_list), 1):
            text = tokenizer.decode([tid], skip_special_tokens=False)
            marker = " <-" if tid == chosen_id else ""
            print(f"    {rank}. {p:7.2%}  id={tid:<7} {text!r}{marker}")
        if chosen_id not in topk_id_list:
            rank = int((probs[0] > chosen_prob).sum().item()) + 1
            print(f"    (sampled token ranked #{rank} in raw distribution) <-")

        tokens = torch.cat([tokens, next_token], dim=1)

        if chosen_id == tokenizer.eos_token_id:
            break

    new_tokens = tokens[0, len(input_ids):].tolist()
    return prompt + tokenizer.decode(new_tokens, skip_special_tokens=False)


def _entropy(probs):
    p = probs[probs > 0]
    return float(-(p * p.log()).sum())


def main():
    parser = argparse.ArgumentParser(
        description="Text generation (greedy or temperature sampling) from a trained MoE checkpoint.",
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the target .safetensors checkpoint to load.",
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", default="The capital of France")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many of the highest-probability tokens to print at each step.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0 = greedy argmax; >0 samples (e.g. 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help=(
            "Keep weights in CPU RAM and stream one module at a time onto the GPU "
            "for the forward pass. The real flash-attn/Triton/scattermoe kernels still "
            "run on CUDA (identical math); only ~one module's weights are GPU-resident, "
            "so peak VRAM ~= the CUDA context + activations. Slow (re-copies weights "
            "every token) but ideal for a smoke test while training holds most of VRAM."
        ),
    )
    parser.add_argument(
        "--max-mem-fraction",
        type=float,
        default=None,
        help=(
            "Cap this process to a fraction of total VRAM (torch.cuda."
            "set_per_process_memory_fraction) so a smoke test fails fast instead of "
            "OOM-ing the live training run. E.g. 0.1 for 10%%."
        ),
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda"

    if args.max_mem_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.max_mem_fraction, 0)

    tokenizer = load_tokenizer(args.tokenizer)
    config = build_config(len(tokenizer), tokenizer.pad_token_id)

    # Older checkpoints from the currently running untied experiment contain a
    # separate output tensor; tied checkpoints omit that duplicate. Detect this
    # before model construction so both generations remain loadable.
    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as f:
        checkpoint_tied = "output_layer.weight" not in f.keys()
    if checkpoint_tied != config.tie_word_embeddings:
        mode = "tied" if checkpoint_tied else "untied"
        print(f"Checkpoint uses {mode} token weights; matching its architecture.")
        config.tie_word_embeddings = checkpoint_tied

    load_device = "cpu" if args.cpu_offload else device

    model = MoEModel(config).to(load_device).eval()
    state_dict = load_file(str(args.checkpoint), device=load_device)
    tied = getattr(model, 'tie_word_embeddings', False)
    model.load_state_dict(state_dict, strict=not tied)
    if tied:
        model.tie_weights()

    if args.cpu_offload:
        from accelerate import cpu_offload
        model = cpu_offload(model, execution_device=device, offload_buffers=False)

    print(generate(model, tokenizer, args.prompt, args.max_new_tokens,
                   args.top_k, args.temperature, device))


if __name__ == "__main__":
    main()
