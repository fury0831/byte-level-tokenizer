"""Render measured stage-4 tables without changing official benchmark metrics.

The official harness owns scoring and 128-token context calculation. This tool
copies those values and audits the reported prefix against per-record counts.
"""

import argparse
import hashlib
import json
from pathlib import Path

from benchmark import read_records
from compare_results import compare


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def record_counts(result):
    rows = [row for category in result["categories"] for row in category["record_results"]]
    counts = {row["id"]: row["tokens"] for row in rows}
    if len(counts) != len(rows):
        raise ValueError("Duplicate per-record IDs in official result")
    return counts


def require_same_model(*results):
    """Equal token counts do not establish a common vocabulary or ID mapping."""
    if len({result["training"]["model_sha256"] for result in results}) != 1:
        raise ValueError("Greedy and DP runs use different models")


def audit_context(result, evaluation):
    """Audit the official retained prefix, never choose a different prefix."""
    context = result["context_budget"]
    counts = record_counts(result)
    if set(counts) != {row["id"] for row in evaluation}:
        raise ValueError("Evaluation IDs differ from the measured results")
    count = context["records"]
    if type(count) is not int or not 0 <= count <= len(evaluation):
        raise ValueError("Invalid official retained record count")
    prefix = evaluation[:count]
    expected = {
        "tokens": sum(counts[row["id"]] for row in prefix),
        "bytes": sum(len(row["text"].encode("utf-8")) for row in prefix),
        "code_points": sum(len(row["text"]) for row in prefix),
    }
    if any(context[key] != value for key, value in expected.items()):
        raise ValueError("Official context does not match the input-order prefix")
    if context["tokens"] > context["budget_tokens"]:
        raise ValueError("Reported context exceeds its budget")
    blocked = evaluation[count] if count < len(evaluation) else None
    next_tokens = counts[blocked["id"]] if blocked else None
    if blocked and context["tokens"] + next_tokens <= context["budget_tokens"]:
        raise ValueError("Reported prefix stopped before a record that still fits")
    return {
        "included_ids": [row["id"] for row in prefix],
        "first_excluded_id": blocked["id"] if blocked else None,
        "first_excluded_tokens": next_tokens,
        "tokens_if_added": context["tokens"] + next_tokens if blocked else None,
    }


def cell(value):
    if value is None:
        return "not produced"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
    ])


def render(greedy_dir, dp_dir, controlled_path, evaluation_path, predictions_path):
    pairs = {}
    lines = ["# Stage 4 measured tables", "",
             "Public smoke data only. Final strategy selection and report are deferred to stage 5.", ""]
    evaluation = read_records(evaluation_path)
    eval_hash = hashlib.sha256(Path(evaluation_path).read_bytes()).hexdigest()
    controlled = load_json(controlled_path)
    predictions = load_json(predictions_path)
    lines += ["## Provenance", "", f"Pre-measurement record: `{predictions_path}`",
              f"Recorded at: {predictions['recorded_at']}. Stage-3 token counts were already known.",
              f"Controlled results: `{controlled_path}`", ""]
    for name, directory in (("greedy", greedy_dir), ("dp", dp_dir)):
        directory = Path(directory)
        summary = load_json(directory / "summary.json")
        if summary["status"] != "passed":
            raise ValueError(f"{directory}: {summary}")
        ref = load_json(directory / "baseline-result.json")
        candidate = load_json(directory / "student-result.json")
        comparison = load_json(directory / "comparison.json")
        if compare(ref, candidate) != comparison:
            raise ValueError("Stored comparison differs from official comparator")
        if candidate["sha256"]["evaluation"] != eval_hash:
            raise ValueError("Evaluation hash mismatch")
        if candidate["training"]["model_sha256"] != ref["training"]["model_sha256"]:
            raise ValueError("This experiment expects the same reused BPE vocabulary")
        pairs[name] = (ref, candidate)
        lines += [f"## End-to-end: baseline vs {name}", "", f"Run: `{directory}`; status: PASS.", "",
                  "Both systems train independently. Training code is shared; equality of models is verified by hash.", ""]
        metrics = [("aggregate", key) for key in ("bytes", "tokens", "bytes_per_token", "encode_seconds", "decode_seconds", "encode_bytes_per_second", "decode_bytes_per_second", "peak_rss_bytes")]
        metrics += [("training", key) for key in ("seconds", "peak_rss_bytes", "vocab_size", "model_bytes")]
        lines += [table(["Metric", "Paired baseline", name],
                        [(f"{section}.{key}", ref[section][key], candidate[section][key]) for section, key in metrics]), ""]
        lines += ["Official total encode/decode seconds are sums of category medians, not medians of whole-dataset times.",
                  "RSS is process high-water memory, including interpreter and validation.", ""]
        refcats = {c["category"]: c for c in ref["categories"]}
        rows = []
        for c in candidate["categories"]:
            r = refcats[c["category"]]
            rows.append([c["category"], c["records"], c["bytes"], r["tokens"], c["tokens"],
                         r["bytes_per_token"], c["bytes_per_token"], r["prepare_seconds"], c["prepare_seconds"],
                         r["encode_seconds"], c["encode_seconds"], r["decode_seconds"], c["decode_seconds"],
                         r["peak_rss_bytes"], c["peak_rss_bytes"]])
        lines += [table(["Category", "Records", "Bytes", "Ref tokens", "Candidate tokens", "Ref B/T", "Candidate B/T", "Ref prep s", "Candidate prep s", "Ref enc s", "Candidate enc s", "Ref dec s", "Candidate dec s", "Ref RSS B", "Candidate RSS B"], rows), "",
                  "Prepare is the median encoding-side preparation; decode uses another fresh runtime.", "",
                  table(["Provisional score field", "Value"], list(comparison.items())), "",
                  "### Environment, configuration and identities", "", "```json",
                  json.dumps({"environment":candidate["environment"], "config":candidate["config"], "sha256":candidate["sha256"], "model_sha256":candidate["training"]["model_sha256"]}, ensure_ascii=False, indent=2), "```", ""]

    ref, dp = pairs["dp"]
    greedy = pairs["greedy"][1]
    require_same_model(ref, dp, greedy, pairs["greedy"][0])
    if greedy["environment"] != dp["environment"] or greedy["config"] != dp["config"]:
        raise ValueError("Greedy and DP runs use different environments or configurations")
    if any(greedy["sha256"][key] != dp["sha256"][key] for key in ("training", "evaluation", "harness")):
        raise ValueError("Greedy and DP runs use different data or harnesses")
    if record_counts(ref) != record_counts(pairs["greedy"][0]):
        raise ValueError("Paired baseline token counts differ")
    if controlled["model"]["file_sha256"] != dp["training"]["model_sha256"]:
        raise ValueError("Controlled model differs from official saved model")
    if controlled["evaluation"]["file_sha256"] != eval_hash:
        raise ValueError("Controlled evaluation differs")
    smoke = controlled["datasets"][0]
    for name, result in (("greedy", greedy), ("dp", dp)):
        measured = {row["id"]:row["tokens"] for row in smoke["results"][name]["records"]}
        if measured != record_counts(result):
            raise ValueError(f"Controlled {name} counts differ from official counts")
    counts = {name:record_counts(result) for name,result in (("baseline",ref),("greedy",greedy),("dp",dp))}
    if any(counts["dp"][r["id"]] > min(counts["greedy"][r["id"]], counts["baseline"][r["id"]]) for r in evaluation):
        raise ValueError("DP violated fixed-vocabulary minimum-token guarantee")
    lines += ["## Per-record compression", "",
              table(["Record", "Category", "UTF-8 bytes", "Code points", "Baseline", "Greedy", "DP", "Greedy minus DP"],
                    [[r["id"], r["category"], len(r["text"].encode()), len(r["text"]), counts["baseline"][r["id"]], counts["greedy"][r["id"]], counts["dp"][r["id"]], counts["greedy"][r["id"]]-counts["dp"][r["id"]]] for r in evaluation]), "",
              "## Controlled comparison", "",
              "One saved model, identical IDs/Trie/decoder and input order; only the encoding policy changes. Whole-dataset medians here differ from the official sum-of-category-medians protocol.", "",
              "Protocol and five raw samples per strategy/dataset are retained in the controlled JSON.", ""]
    keys = ["median_preparation_encode_seconds", "median_preparation_decode_seconds", "median_encode_seconds", "median_decode_seconds", "encode_bytes_per_second", "decode_bytes_per_second", "median_peak_rss_bytes", "max_peak_rss_bytes"]
    lines += [table(["Whole smoke dataset", "Greedy", "DP"], [(key, smoke["results"]["greedy"]["summary"][key], smoke["results"]["dp"]["summary"][key]) for key in keys]), "",
              "### Supplementary synthetic length sweep", "",
              "No synthetic data is used for training. Actual UTF-8 length is shown; Unicode is never byte-truncated. RSS cannot isolate DP arrays.", ""]
    rows = []
    for dataset in controlled["datasets"][1:]:
        g, d = (dataset["results"][p] for p in ("greedy", "dp"))
        rows.append([dataset["name"], g["totals"]["bytes"], g["totals"]["tokens"], d["totals"]["tokens"], g["summary"]["median_encode_seconds"], d["summary"]["median_encode_seconds"], d["summary"]["median_encode_seconds"]/g["summary"]["median_encode_seconds"], g["summary"]["median_peak_rss_bytes"], d["summary"]["median_peak_rss_bytes"]])
    lines += [table(["Dataset", "Actual bytes", "G tokens", "DP tokens", "G enc s", "DP enc s", "DP/G time", "G median RSS B", "DP median RSS B"], rows), "",
              "## Official 128-token whole-record context", "",
              "Original evaluation-file order; independent record encoding; stop before the first record that does not fit. No truncation, skipping, cross-record merge or special tokens.", ""]
    models = {"baseline":ref, "greedy":greedy, "dp":dp}
    audits = {name:audit_context(result,evaluation) for name,result in models.items()}
    lines += [table(["Copied official field", *models], [(key, *(r["context_budget"][key] for r in models.values())) for key in ("budget_tokens","records","bytes","code_points","tokens")]), "",
              table(["Boundary audit", *models], [(key, *(a[key] for a in audits.values())) for key in ("first_excluded_id","first_excluded_tokens","tokens_if_added")]), "",
              "Included prefix IDs: " + ", ".join(f"{name}: {audit['included_ids']}" for name,audit in audits.items()), "",
              "The retained values above come directly from context_budget. The boundary audit independently checks consistency with per-record counts.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--greedy-run", required=True)
    parser.add_argument("--dp-run", required=True)
    parser.add_argument("--controlled", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output exists; choose a new path")
    content = render(args.greedy_run,args.dp_run,args.controlled,args.evaluation,args.predictions)
    with output.open("x",encoding="utf-8") as handle:
        handle.write(content)
    print(f"Saved measured tables: {output}")


if __name__ == "__main__":
    main()
