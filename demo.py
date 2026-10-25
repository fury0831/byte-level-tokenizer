"""Compare segmentation policies with an explicit vocabulary; no training."""

import argparse
import json
from pathlib import Path

from tokenizer_core import build_runtime, decode_strict, encode_dp, encode_greedy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="abcd", help="Text to segment, without normalization")
    parser.add_argument("--strategy", choices=("both", "greedy", "dp"), default="both")
    parser.add_argument(
        "--model", type=Path,
        default=Path(__file__).resolve().parent / "examples" / "toy-vocabulary.json",
        help="Explicit vocabulary JSON; no model is downloaded or trained",
    )
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    runtime = build_runtime(model)
    encoders = {"greedy": encode_greedy, "dp": encode_dp}
    selected = encoders if args.strategy == "both" else {args.strategy: encoders[args.strategy]}
    results = {}
    for name, encode in selected.items():
        ids = encode(args.text, runtime)
        decoded = decode_strict(ids, runtime)
        results[name] = {
            "token_ids": ids,
            "token_hex": [runtime.tokens[i].hex() for i in ids],
            "token_count": len(ids),
            "roundtrip_equal": decoded == args.text,
        }
    print(json.dumps({"text": args.text, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
