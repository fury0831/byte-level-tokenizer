"""Course integration implementation retained for private review.

Original exercise/setup and contract prose has been omitted. The original
function signatures and implementation statements remain; all training and
checker behavior still requires separately authorized course components.
This file is not a replacement trainer or a self-contained course solution.
"""

from baseline_bpe import train as _baseline_train
from tokenizer_core import build_runtime, decode_strict, encode_dp

def train(records: list[str], vocab_size: int, seed: int) -> dict:
    return _baseline_train(records, vocab_size, seed)

def prepare(model: dict) -> object:
    return build_runtime(model)

def encode(text: str, runtime: object) -> list[int]:
    return encode_dp(text, runtime)

def decode(ids: list[int], runtime: object) -> str:
    return decode_strict(ids, runtime)
if __name__ == '__main__':
    from project1_check import main
    raise SystemExit(main(default_adapter=__file__))
