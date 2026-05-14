from pathlib import Path

from transformers import PreTrainedTokenizerFast


TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "tokenizer.json"
EOS_TOKEN = "<|endoftext|>"


def load_tokenizer():
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_PATH),
        eos_token=EOS_TOKEN,
        pad_token=EOS_TOKEN,
    )

    return tokenizer
