"""Controlled greedy/DP comparison on one saved vocabulary, without training.

Run from src with ``python -m experiments.compare_encoders --help``. Every
strategy/pass runs sequentially in a fresh child process. Timings exclude
startup, model/data I/O, warm-up, and validation. RSS is the whole child's
high-water mark, including those costs; it is not encoder-only allocation.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import shlex
import statistics
import subprocess
import sys
import time

from tokenizer_core import (
    build_runtime, decode_strict, encode_dp, encode_greedy,
)


SOURCE_ROOT = Path(__file__).resolve().parents[1]
ENCODERS = {"greedy": encode_greedy, "dp": encode_dp}
TARGET_BYTES = (64, 512, 4096, 32768)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_records(records):
    """Validate exact input strings and unambiguous record identities."""
    if not isinstance(records, list) or not records:
        raise ValueError("Dataset must contain at least one record")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each record must be a JSON object")
        for field in ("id", "category", "text"):
            if not isinstance(record.get(field), str):
                raise ValueError(f"Record {field} must be a string")
        if not record["id"] or record["id"] in seen:
            raise ValueError("Record IDs must be nonempty and unique")
        if not record["category"]:
            raise ValueError("Record category must be nonempty")
        seen.add(record["id"])
        record["text"].encode("utf-8", errors="strict")
    return records


def load_records(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    return validate_records(records)


def record_metadata(record, include_text=False):
    text = record["text"]
    encoded = text.encode("utf-8")
    result = {"id": record["id"], "category": record["category"],
              "bytes": len(encoded), "codepoints": len(text),
              "text_sha256": digest(encoded)}
    if include_text:
        result["text"] = text
    return result


def make_synthetic_datasets(seed):
    """Construct inputs independently of evaluation contents, before timing."""
    units = {
        "ascii_repeat": "ab cd  ",
        "chinese_repeat": "中文测试。",
        "code_repeat": "for i in range(10):\n    total += i\n",
    }
    datasets = []
    # One fixed random stream, with nested prefixes at increasing sizes.
    generator = random.Random(seed)
    entropy = "".join(chr(generator.randrange(32, 127)) for _ in range(max(TARGET_BYTES)))
    for family in (*units, "high_entropy_ascii"):
        for target in TARGET_BYTES:
            if family in units:
                unit = units[family]
                repetitions = (target + len(unit.encode("utf-8")) - 1) // len(unit.encode("utf-8"))
                text = unit * repetitions
                construction = {"unit": unit, "repetitions": repetitions,
                                "rule": "repeat whole unit until UTF-8 bytes >= target"}
            else:
                text = entropy[:target]
                construction = {"seed": seed, "generator": "random.Random(seed).randrange(32, 127)",
                                "rule": "prefix of one 32768-character printable-ASCII stream"}
            name = f"{family}-{target}"
            construction["target_bytes"] = target
            datasets.append({"name": name, "kind": "synthetic_length_sweep",
                             "construction": construction,
                             "records": [{"id": name, "category": family, "text": text}]})
    return datasets


def validate_encoded(record, ids, runtime):
    if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < len(runtime.tokens)
                                        for i in ids):
        raise ValueError("Encoder returned invalid token IDs")
    if b"".join(runtime.tokens[i] for i in ids) != record["text"].encode("utf-8"):
        raise ValueError(f"Byte reconstruction failed for {record['id']}")
    if decode_strict(ids, runtime) != record["text"]:
        raise ValueError(f"Round-trip failed for {record['id']}")


def totals(rows):
    result = {"records": len(rows)}
    for field in ("bytes", "codepoints", "tokens"):
        result[field] = sum(row[field] for row in rows)
    result["bytes_per_token"] = result["bytes"] / result["tokens"] if result["tokens"] else None
    return result


def summarize_passes(passes):
    """Aggregate only verified passes; timings remain available individually."""
    if not passes:
        raise ValueError("At least one pass is required")
    rows = passes[0]["records"]
    if any(result["records"] != rows for result in passes[1:]):
        raise ValueError("Tokenization changed between fresh-process passes")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    aggregate = totals(rows)
    summary = {}
    timing_fields = ("preparation_encode_seconds", "preparation_decode_seconds",
                     "encode_seconds", "decode_seconds")
    for field in timing_fields:
        summary[f"median_{field}"] = statistics.median(result[field] for result in passes)
    for operation in ("encode", "decode"):
        elapsed = summary[f"median_{operation}_seconds"]
        summary[f"{operation}_bytes_per_second"] = aggregate["bytes"] / elapsed if elapsed else None
    summary["median_peak_rss_bytes"] = statistics.median(result["peak_rss_bytes"] for result in passes)
    summary["max_peak_rss_bytes"] = max(result["peak_rss_bytes"] for result in passes)
    return {"records": rows, "totals": aggregate,
            "by_category": {category: totals(items) for category, items in sorted(grouped.items())},
            "passes": [{key: value for key, value in result.items() if key != "records"}
                       for result in passes], "summary": summary}


def measure_worker(request):
    """Worker operation; all correctness checks are outside measured intervals."""
    model = request["model"]
    original = canonical_bytes(model)
    if digest(original) != request["model_sha256"]:
        raise ValueError("Worker model hash differs from frozen model")
    records = validate_records(request["records"])
    encoder = ENCODERS[request["strategy"]]

    # A complete untimed warm-up in each isolated process, then fresh states.
    warm_runtime = build_runtime(model)
    for record in records:
        warm_ids = encoder(record["text"], warm_runtime)
        validate_encoded(record, warm_ids, warm_runtime)
    del warm_runtime

    start = time.perf_counter()
    encoding_runtime = build_runtime(model)
    preparation_encode = time.perf_counter() - start
    start = time.perf_counter()
    encoded = [encoder(record["text"], encoding_runtime) for record in records]
    encode_seconds = time.perf_counter() - start
    for record, ids in zip(records, encoded):
        validate_encoded(record, ids, encoding_runtime)

    start = time.perf_counter()
    decoding_runtime = build_runtime(model)
    preparation_decode = time.perf_counter() - start
    start = time.perf_counter()
    decoded = [decode_strict(ids, decoding_runtime) for ids in encoded]
    decode_seconds = time.perf_counter() - start
    if decoded != [record["text"] for record in records]:
        raise ValueError("Timed decode returned different text")

    # Both repeated use and fresh/reversed order must preserve exact IDs.
    for record, ids in zip(records, encoded):
        if encoder(record["text"], encoding_runtime) != ids:
            raise ValueError("Repeated encoding is not deterministic")
    reverse_runtime = build_runtime(model)
    for record, ids in zip(reversed(records), reversed(encoded)):
        if encoder(record["text"], reverse_runtime) != ids:
            raise ValueError("Encoding depends on record order")
    if canonical_bytes(model) != original:
        raise ValueError("Encoder modified the frozen saved model")

    rows = []
    for record, ids in zip(records, encoded):
        row = record_metadata(record)
        row.update(tokens=len(ids), ids_sha256=digest(canonical_bytes(ids)))
        rows.append(row)
    # macOS reports bytes; Linux reports KiB. The supported environments are
    # Linux/macOS (including WSL). Peak includes warm-up and validation.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    return {"index": request["index"], "preparation_encode_seconds": preparation_encode,
            "preparation_decode_seconds": preparation_decode,
            "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
            "peak_rss_bytes": rss_bytes, "records": rows}


def run_child(request):
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED=str(request["seed"]), OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "experiments.compare_encoders", "--worker"],
        input=canonical_bytes(request), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=SOURCE_ROOT, env=environment, timeout=120, check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    return json.loads(completed.stdout)


def compare(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1")
    model_file = args.model.read_bytes()
    model = json.loads(model_file)
    runtime = build_runtime(model)
    fixed_hash = digest(canonical_bytes(model))
    evaluation_file = args.evaluation.read_bytes()
    evaluation = load_records(args.evaluation)
    datasets = [{"name": "smoke_eval", "kind": "supplied_evaluation",
                 "construction": {"source": str(args.evaluation.resolve()), "record_order": "file order"},
                 "records": evaluation}, *make_synthetic_datasets(args.seed)]
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, "-m", "experiments.compare_encoders", "--model",
                               str(args.model), "--evaluation", str(args.evaluation), "--repeats",
                               str(args.repeats), "--seed", str(args.seed), "--output", str(args.output)]),
        "working_directory": str(Path.cwd()),
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "machine": platform.machine()},
        "protocol": {
            "repeats": args.repeats, "seed": args.seed, "training": "none; one frozen saved model",
            "fixed": ["model", "vocabulary", "ID table", "byte Trie", "decoder", "records", "token cost = 1"],
            "changed": "greedy longest match versus DP minimum-token segmentation",
            "strategy_order": [list(ENCODERS) if index % 2 == 0 else list(reversed(ENCODERS))
                               for index in range(args.repeats)],
            "processes": "one fresh child per dataset/strategy/pass; all children sequential",
            "threads": "one computational thread; standard library only",
            "warmup": "one full encode/decode pass per child before timing",
            "timing": "perf_counter; fresh encoding and decoding runtimes; preparation measured separately",
            "excluded_from_timing": ["process startup", "model/data I/O", "warmup", "validation"],
            "validation": ["valid IDs", "exact byte reconstruction", "strict UTF-8 round trip",
                           "repeated-call determinism", "fresh-runtime reversed record order", "unchanged model"],
            "rss": "whole-child peak RSS including interpreter, input, warmup, validation, and retained outputs; not encoder-only",
            "aggregation": "whole-dataset median times, bytes/median seconds; per-category token counts only",
            "synthetic": "four predefined families and four target lengths, independent of evaluation content; no training",
            "prediction": "DP uses no more tokens per record than greedy; DP may cost more time and memory",
        },
        "sources_sha256": {str(path.relative_to(SOURCE_ROOT)): digest(path.read_bytes()) for path in
                           (Path(__file__), SOURCE_ROOT / "tokenizer_core.py")},
        "model": {"path": str(args.model.resolve()), "file_sha256": digest(model_file),
                  "canonical_sha256": fixed_hash, "vocab_size": len(runtime.tokens), "file_bytes": len(model_file)},
        "evaluation": {"path": str(args.evaluation.resolve()), "file_sha256": digest(evaluation_file)},
        "datasets": [],
    }
    for dataset in datasets:
        passes = {strategy: [] for strategy in ENCODERS}
        for index, order in enumerate(result["protocol"]["strategy_order"]):
            for strategy in order:
                passes[strategy].append(run_child({"model": model, "model_sha256": fixed_hash,
                                                  "records": dataset["records"], "strategy": strategy,
                                                  "index": index + 1, "seed": args.seed}))
        summaries = {strategy: summarize_passes(items) for strategy, items in passes.items()}
        for greedy, dp in zip(summaries["greedy"]["records"], summaries["dp"]["records"]):
            if dp["tokens"] > greedy["tokens"]:
                raise ValueError(f"DP optimality condition failed for {dp['id']}")
        result["datasets"].append({
            "name": dataset["name"], "kind": dataset["kind"], "construction": dataset["construction"],
            "records": [record_metadata(record, include_text=dataset["kind"] != "supplied_evaluation")
                        for record in dataset["records"]],
            "sha256": digest(canonical_bytes(dataset["records"])), "results": summaries,
            "dp_token_savings": summaries["greedy"]["totals"]["tokens"] - summaries["dp"]["totals"]["tokens"],
        })
        print(f"{dataset['name']}: greedy={summaries['greedy']['totals']['tokens']}, "
              f"dp={summaries['dp']['totals']['tokens']}", file=sys.stderr, flush=True)
    if args.model.read_bytes() != model_file or args.evaluation.read_bytes() != evaluation_file:
        raise ValueError("Model or evaluation file changed during measurement")
    for name, expected in result["sources_sha256"].items():
        if digest((SOURCE_ROOT / name).read_bytes()) != expected:
            raise ValueError(f"Source changed during measurement: {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=True, allow_nan=False, indent=2)
        output.write("\n")
    return result


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if arguments == ["--worker"]:
        json.dump(measure_worker(json.load(sys.stdin)), sys.stdout, allow_nan=False)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=5051)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    try:
        compare(args)
    except (ValueError, TypeError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Comparison failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
