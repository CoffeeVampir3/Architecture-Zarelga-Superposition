import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

from modeling.model import MoEModel
from modeling.model_config import ModelConfig


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "checkpoint_epoch_1.safetensors"
DEFAULT_TOKENIZER = ROOT / "tokenizer.json"
EOS_TOKEN = "<|endoftext|>"


def load_tokenizer(tokenizer_path: Path):
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        eos_token=EOS_TOKEN,
        pad_token=EOS_TOKEN,
    )


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = torch.tensor([input_ids], dtype=torch.long, device="cuda")

    for _ in range(max_new_tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(tokens)

        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    new_tokens = tokens[0, len(input_ids):].tolist()
    return prompt + tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", default="She")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    config = ModelConfig(vocab_size=len(tokenizer))

    model = MoEModel(config).cuda().eval()
    state_dict = load_file(str(args.checkpoint), device="cuda")
    model.load_state_dict(state_dict)

    print(generate(model, tokenizer, args.prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()
