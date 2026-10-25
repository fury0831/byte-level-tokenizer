"""Minimum-token candidate using the course BPE vocabulary with attribution.

This adapter shares its vocabulary, Trie and decoder with adapter_greedy;
only the encoding policy differs. Optimality is for a fixed vocabulary.
"""

from baseline_bpe import train
from tokenizer_core import build_runtime as prepare
from tokenizer_core import decode_strict as decode
from tokenizer_core import encode_dp as encode


if __name__ == "__main__":
    from project1_check import main

    raise SystemExit(main(default_adapter=__file__))
