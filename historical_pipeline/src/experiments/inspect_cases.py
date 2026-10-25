"""Save untimed, frozen-model token traces for explicitly selected records."""

import argparse
import hashlib
import json
from pathlib import Path

import baseline_bpe
from experiments.inspect_tokens import trace_tokens
from tokenizer_core import build_runtime, encode_dp, encode_greedy


def select_requested_records(records, requested_ids):
    """Select in CLI order; reject missing IDs or ambiguous duplicate records."""
    if not requested_ids or len(set(requested_ids)) != len(requested_ids):
        raise ValueError("Request at least one distinct record ID")
    by_id = {}
    for record in records:
        if not isinstance(record, dict) or not all(
            isinstance(record.get(key), str) for key in ("id", "category", "text")
        ):
            raise ValueError("Each evaluation record needs string id/category/text")
        if record["id"] in by_id:
            raise ValueError("Duplicate evaluation record ID: " + record["id"])
        by_id[record["id"]] = record
    missing = [record_id for record_id in requested_ids if record_id not in by_id]
    if missing:
        raise ValueError("Unknown record IDs: " + ", ".join(missing))
    return [by_id[record_id] for record_id in requested_ids]


def canonical_model_bytes(model):
    return json.dumps(model, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def file_fingerprint(path, payload=None):
    path = Path(path).resolve()
    if payload is None:
        payload = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def inspect_cases(model_path, evaluation_path, requested_ids):
    model_path, evaluation_path = Path(model_path).resolve(), Path(evaluation_path).resolve()
    model_bytes, evaluation_bytes = model_path.read_bytes(), evaluation_path.read_bytes()
    model = json.loads(model_bytes)
    original_model = canonical_model_bytes(model)
    runtime = build_runtime(model)
    bpe_runtime = baseline_bpe.prepare(model)
    records = [json.loads(line) for line in evaluation_bytes.decode("utf-8").splitlines()
               if line.strip()]
    selected = select_requested_records(records, requested_ids)
    src = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve(), src / "experiments" / "inspect_tokens.py",
                    src / "tokenizer_core.py", src / "baseline_bpe.py"]
    result = {
        "schema_version": 1,
        "experiment": "untimed-frozen-model-record-traces",
        "model": {**file_fingerprint(model_path, model_bytes),
                  "canonical_sha256": hashlib.sha256(original_model).hexdigest(),
                  "vocabulary_size": len(runtime.tokens)},
        "evaluation": file_fingerprint(evaluation_path, evaluation_bytes),
        "sources": [file_fingerprint(path) for path in source_paths],
        "requested_record_ids": list(requested_ids),
        "offset_convention": "zero-based UTF-8 byte offsets; exclusive end",
        "records": [],
    }
    for record in selected:
        text = record["text"]
        data = text.encode("utf-8", errors="strict")
        row = {"id": record["id"], "category": record["category"], "text": text,
               "utf8_hex": data.hex(), "bytes": len(data), "code_points": len(text),
               "strategies": {}}
        encoded = {
            "bpe": baseline_bpe.encode(text, bpe_runtime),
            "greedy": encode_greedy(text, runtime),
            "dp": encode_dp(text, runtime),
        }
        for name, ids in encoded.items():
            token_rows = trace_tokens(text, ids, runtime)
            row["strategies"][name] = {"token_count": len(ids), "ids": ids,
                                        "tokens": token_rows, "roundtrip_valid": True}
        result["records"].append(row)
    if canonical_model_bytes(model) != original_model:
        raise ValueError("The frozen model changed while creating traces")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--record-id", required=True, action="append", dest="record_ids")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".json":
        parser.error("--output must have a .json suffix")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output")
    result = inspect_cases(args.model, args.evaluation, args.record_ids)
    payload = json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(payload)
    print(f"Saved untimed record traces: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
