"""Longest-match candidate using the course BPE vocabulary with attribution.

The unmodified baseline_bpe module supplies training; tokenizer_core supplies
the shared immutable Trie and strict decoder. No separate learned state.
"""

from baseline_bpe import train
from tokenizer_core import build_runtime as prepare
from tokenizer_core import decode_strict as decode
from tokenizer_core import encode_greedy as encode


if __name__ == "__main__":
    from project1_check import main

    raise SystemExit(main(default_adapter=__file__))
