"""Produce reproducible token/embedding traces from a frozen saved model.

Run from src with ``python -m experiments.inspect_tokens --help``.
No training, model mutation, or language-model calls take place here.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path

from tokenizer_core import build_runtime, decode_strict, encode_dp, encode_greedy


CATEGORIES = ("english", "chinese", "code", "unicode")
ENCODERS = {"greedy": encode_greedy, "dp": encode_dp}


def escaped_text(text):
    """A complete JSON string, with controls/Unicode/fence markers escaped."""
    return json.dumps(text, ensure_ascii=True).replace("`", "\\u0060")


def table_code(value):
    """Protect Markdown table delimiters and HTML even for hostile input."""
    return "<code>" + html.escape(str(value)).replace("|", "&#124;").replace("`", "&#96;") + "</code>"


def trace_tokens(text, ids, runtime):
    """Validate reconstruction and expose contiguous, half-open byte spans."""
    reconstructed = decode_strict(ids, runtime)
    if reconstructed != text:
        raise ValueError("Token IDs do not reconstruct the supplied text")
    original = text.encode("utf-8", errors="strict")
    rows = []
    start = 0
    for index, token_id in enumerate(ids):
        token = runtime.tokens[token_id]
        end = start + len(token)
        if original[start:end] != token:
            raise ValueError("Token byte span does not match the input")
        try:
            display = escaped_text(token.decode("utf-8", errors="strict"))
        except UnicodeDecodeError:
            display = "partial UTF-8"
        rows.append({"index": index, "start": start, "end": end,
                     "id": token_id, "hex": token.hex(), "display": display})
        start = end
    if start != len(original):
        raise ValueError("Token spans do not cover the input bytes")
    return rows


def select_records(records):
    """Shortest nonempty UTF-8 record per category; ties use source order."""
    selected = {}
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict) or not all(
            isinstance(record.get(key), str) for key in ("id", "category", "text")
        ):
            raise ValueError("Each evaluation record needs string id/category/text")
        text = record["text"]
        byte_length = len(text.encode("utf-8", errors="strict"))
        category = record["category"]
        if category not in CATEGORIES or not text:
            continue
        key = (byte_length, position)
        if category not in selected or key < selected[category][0]:
            selected[category] = (key, record)
    missing = [category for category in CATEGORIES if category not in selected]
    if missing:
        raise ValueError("Missing nonempty example categories: " + ", ".join(missing))
    return [(category, selected[category][0][1], selected[category][1])
            for category in CATEGORIES]


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def render_report(model_path, evaluation_path, strategy="both"):
    """Return Markdown with full real examples and independently checked spans."""
    if strategy not in ("both", *ENCODERS):
        raise ValueError("Unknown strategy")
    model_path = Path(model_path).resolve()
    evaluation_path = Path(evaluation_path).resolve()
    model_bytes = model_path.read_bytes()
    evaluation_bytes = evaluation_path.read_bytes()
    model = json.loads(model_bytes)
    runtime = build_runtime(model)
    canonical = json.dumps(model, sort_keys=True, ensure_ascii=True,
                           allow_nan=False, separators=(",", ":")).encode("utf-8")
    records = [json.loads(line) for line in evaluation_bytes.decode("utf-8").splitlines()
               if line.strip()]
    examples = select_records(records)
    strategies = tuple(ENCODERS) if strategy == "both" else (strategy,)
    src = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), src / "tokenizer_core.py"]
    lines = [
        "# Frozen-vocabulary token traces", "",
        "These are actual encodings of the shortest nonempty UTF-8 record in each "
        "requested category; ties retain original evaluation order. All records are "
        "encoded independently. Record positions below are one-based; token indices "
        "and byte offsets are zero-based, with exclusive end offsets.", "",
        "| Input or implementation | Exact path | SHA-256 of file bytes |",
        "| --- | --- | --- |",
        f"| Saved model | {table_code(model_path)} | {table_code(sha256(model_bytes))} |",
        f"| Evaluation | {table_code(evaluation_path)} | {table_code(sha256(evaluation_bytes))} |",
    ]
    for source in sources:
        lines.append(f"| Source | {table_code(source)} | {table_code(sha256(source.read_bytes()))} |")
    lines += ["", f"Canonical model SHA-256: `{sha256(canonical)}`. "
              f"Vocabulary size: {len(runtime.tokens)}. Strategies: {', '.join(strategies)}.", "",
              "Text and valid token displays use JSON escapes, preserving whitespace, "
              "normalization, control characters, and literal Markdown. A token that "
              "cannot itself decode strictly is labelled **partial UTF-8**; its hex "
              "bytes are authoritative. No replacement decoding is used.", ""]
    for category, position, record in examples:
        text = record["text"]
        data = text.encode("utf-8", errors="strict")
        lines += [f"## {category}", "",
                  f"Record ID: {table_code(escaped_text(record['id']))}; "
                  f"evaluation position: {position}; {len(data)} bytes; "
                  f"{len(text)} Unicode code points.", "",
                  "Complete text (JSON string):", "", "```json", escaped_text(text), "```", "",
                  "UTF-8 bytes:", "", "```text", data.hex(" "), "```", ""]
        for name in strategies:
            ids = ENCODERS[name](text, runtime)
            rows = trace_tokens(text, ids, runtime)
            lines += [f"### {name}", "", f"Ordered IDs ({len(ids)} tokens):", "",
                      "```json", json.dumps(ids), "```", "",
                      "| Index | Byte span [start, end) | Token ID | Hex bytes | Strict token display |",
                      "| ---: | --- | ---: | --- | --- |"]
            for row in rows:
                lines.append(f"| {row['index']} | [{row['start']}, {row['end']}) | "
                             f"{row['id']} | {table_code(row['hex'])} | "
                             f"{table_code(row['display'])} |")
            lines += ["", "Validation: concatenated token bytes equal the original UTF-8 bytes; "
                      "strict decoding reconstructs the complete text.", "",
                      "Conceptual embedding rows, in sequence order:", "", "```text",
                      " -> ".join(f"E[{token_id}]" for token_id in ids), "```", ""]
    lines += [
        "## Connection to an AI model", "",
        "The demonstrated path is text → UTF-8 bytes → vocabulary tokens → integer IDs "
        "→ embedding lookup E[id] → model. E is a hypothetical learned embedding table; "
        "no embedding values or model outputs were computed or invented. Every row must "
        "retain the token meaning associated with that ID during language-model training. "
        "Reordering IDs without correspondingly remapping the model changes those meanings.", "",
        "Greedy and DP use the same ID table here, but may feed different token sequences "
        "to a language model. Minimum token count under this vocabulary does not prove "
        "better language-model quality. Fewer tokens can fit more text in a fixed context "
        "budget and reduce sequence-processing work; a larger vocabulary increases the "
        "embedding and output-head dimensions for a fixed hidden width. To assess a changed "
        "segmentation policy, a later controlled language-model experiment would train or "
        "adapt models consistently to each policy, control data/compute and architecture, "
        "and compare held-out loss normalized per byte (or another shared text unit), "
        "downstream quality, and compute. Raw per-token perplexities across different "
        "segmentations are not directly comparable. This artifact only measures tokenization.", "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--strategy", choices=("dp", "greedy", "both"), default="both")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".md":
        parser.error("--output must have a .md suffix")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output")
    report = render_report(args.model, args.evaluation, args.strategy)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(report)
    print(f"Saved token traces: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
