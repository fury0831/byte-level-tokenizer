"""Independent checks for experiment bookkeeping, inputs, and isolation setup.

No timing assertions or performance measurements are made by these tests.
"""

import argparse
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import compare_encoders as experiment
from tokenizer_core import MODEL_FORMAT, build_runtime, encode_greedy


def fixture_model():
    return {"format": MODEL_FORMAT,
            "tokens_hex": [bytes([i]).hex() for i in range(256)] + [b"ab".hex(), b"bcd".hex()]}


def fixture_pass(index=1):
    return {
        "index": index, "preparation_encode_seconds": 0.1 * index,
        "preparation_decode_seconds": 0.2 * index,
        "encode_seconds": float(index), "decode_seconds": 2.0 * index,
        "peak_rss_bytes": index * 100,
        "records": [
            {"id": "a", "category": "english", "bytes": 4, "codepoints": 4,
             "tokens": 2, "ids_sha256": "first"},
            {"id": "b", "category": "chinese", "bytes": 6, "codepoints": 2,
             "tokens": 3, "ids_sha256": "second"},
            {"id": "c", "category": "english", "bytes": 0, "codepoints": 0,
             "tokens": 0, "ids_sha256": "empty"},
        ],
    }


class DatasetTests(unittest.TestCase):
    def test_utf8_byte_and_codepoint_counts_are_distinct(self):
        record = {"id": "unicode", "category": "mixed", "text": "你🙂e\u0301"}
        row = experiment.record_metadata(record)
        self.assertEqual(row["bytes"], 10)
        self.assertEqual(row["codepoints"], 4)
        self.assertEqual(row["text_sha256"], experiment.digest(record["text"].encode("utf-8")))
        self.assertNotIn("text", row)
        self.assertEqual(experiment.record_metadata(record, True)["text"], record["text"])

    def test_dataset_rejects_missing_fields_duplicate_ids_and_invalid_unicode(self):
        valid = {"id": "one", "category": "x", "text": ""}
        self.assertEqual(experiment.validate_records([valid]), [valid])
        invalid = [[], {}, [valid, valid], [{"id": "one", "category": "x"}],
                   [{"id": "", "category": "x", "text": "ok"}],
                   [{"id": "one", "category": "", "text": "ok"}],
                   [{"id": "one", "category": "x", "text": 4}],
                   [{"id": "one", "category": "x", "text": "\ud800"}]]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises((ValueError, UnicodeError)):
                experiment.validate_records(records)

    def test_jsonl_preserves_whitespace_and_record_order(self):
        records = [{"id": "second", "category": "code", "text": "  x\t\n"},
                   {"id": "first", "category": "unicode", "text": "e\u0301"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in records) + "\n\n", encoding="utf-8")
            self.assertEqual(experiment.load_records(path), records)

    def test_synthetic_shapes_are_fixed_and_repeat_whole_units(self):
        datasets = experiment.make_synthetic_datasets(5051)
        self.assertEqual(len(datasets), 16)
        self.assertEqual(len({row["name"] for row in datasets}), 16)
        for dataset in datasets:
            record = dataset["records"][0]
            construction = dataset["construction"]
            actual = len(record["text"].encode("utf-8"))
            self.assertGreaterEqual(actual, construction["target_bytes"])
            self.assertEqual(dataset["kind"], "synthetic_length_sweep")
            if "unit" in construction:
                self.assertEqual(record["text"], construction["unit"] * construction["repetitions"])
                self.assertLess(actual, construction["target_bytes"] + len(construction["unit"].encode("utf-8")))
            else:
                self.assertEqual(actual, construction["target_bytes"])
                self.assertTrue(all(32 <= ord(character) <= 126 for character in record["text"]))

    def test_synthetic_randomness_is_reproducible_and_uses_nested_prefixes(self):
        first = experiment.make_synthetic_datasets(5051)
        self.assertEqual(first, experiment.make_synthetic_datasets(5051))
        changed = experiment.make_synthetic_datasets(5052)
        self.assertEqual(first[:12], changed[:12])
        self.assertNotEqual(first[12:], changed[12:])
        random_texts = [dataset["records"][0]["text"] for dataset in first[12:]]
        for short, long in zip(random_texts, random_texts[1:]):
            self.assertTrue(long.startswith(short))


class SummaryTests(unittest.TestCase):
    def test_counts_are_summed_and_ratio_is_not_average_of_ratios(self):
        summary = experiment.summarize_passes([fixture_pass(i) for i in (5, 1, 4, 2, 3)])
        self.assertEqual(summary["totals"], {"records": 3, "bytes": 10, "codepoints": 6,
                                            "tokens": 5, "bytes_per_token": 2.0})
        self.assertEqual(summary["by_category"]["english"]["records"], 2)
        self.assertEqual(summary["by_category"]["chinese"]["codepoints"], 2)
        self.assertEqual(summary["summary"]["median_encode_seconds"], 3.0)
        self.assertEqual(summary["summary"]["median_decode_seconds"], 6.0)
        self.assertAlmostEqual(summary["summary"]["encode_bytes_per_second"], 10 / 3)
        self.assertEqual(summary["summary"]["median_peak_rss_bytes"], 300)
        self.assertEqual(summary["summary"]["max_peak_rss_bytes"], 500)
        self.assertEqual(len(summary["passes"]), 5)
        self.assertNotIn("records", summary["passes"][0])

    def test_pass_agreement_compares_exact_ids_hash_not_only_token_count(self):
        first, second = fixture_pass(), fixture_pass(2)
        second["records"][0]["ids_sha256"] = "different-segmentation-same-count"
        with self.assertRaisesRegex(ValueError, "changed"):
            experiment.summarize_passes([first, second])
        with self.assertRaises(ValueError):
            experiment.summarize_passes([])

    def test_empty_text_ratio_is_undefined_not_fabricated_zero(self):
        self.assertIsNone(experiment.totals([{"bytes": 0, "codepoints": 0, "tokens": 0}])["bytes_per_token"])

    def test_validation_catches_wrong_but_in_range_ids(self):
        record = {"id": "case", "category": "x", "text": "abcd"}
        runtime = build_runtime(fixture_model())
        experiment.validate_encoded(record, encode_greedy(record["text"], runtime), runtime)
        for ids in ([256, 100], [True], [-1], [258], [1.0]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                experiment.validate_encoded(record, ids, runtime)


class OrchestrationTests(unittest.TestCase):
    def test_existing_output_and_invalid_repeats_fail_before_any_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("do not overwrite", encoding="utf-8")
            args = argparse.Namespace(output=output, model=Path("missing"), evaluation=Path("missing"), repeats=5, seed=5051)
            with patch.object(experiment, "run_child") as child:
                with self.assertRaises(FileExistsError):
                    experiment.compare(args)
                self.assertEqual(output.read_text(encoding="utf-8"), "do not overwrite")
                args.output = Path(directory) / "new.json"
                args.repeats = 0
                with self.assertRaises(ValueError):
                    experiment.compare(args)
                child.assert_not_called()

    def test_frozen_model_alternating_order_and_roundtrip_output(self):
        # Replace all measurement with deterministic hand-specified durations.
        # The actual encoders only supply the expected record bookkeeping.
        requests = []

        def fake_child(request):
            requests.append(copy.deepcopy(request))
            runtime = build_runtime(request["model"])
            rows = []
            for record in request["records"]:
                ids = experiment.ENCODERS[request["strategy"]](record["text"], runtime)
                row = experiment.record_metadata(record)
                row.update(tokens=len(ids), ids_sha256=experiment.digest(experiment.canonical_bytes(ids)))
                rows.append(row)
            result = fixture_pass(request["index"])
            result["records"] = rows
            return result

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model_path, data_path, output = directory / "model.json", directory / "eval.jsonl", directory / "result.json"
            model = fixture_model()
            model_path.write_bytes(experiment.canonical_bytes(model))
            data_path.write_text(json.dumps({"id": "a", "category": "english", "text": "abcd"}) + "\n", encoding="utf-8")
            args = argparse.Namespace(model=model_path, evaluation=data_path, output=output, repeats=5, seed=5051)
            with patch.object(experiment, "run_child", side_effect=fake_child), \
                 patch.object(experiment, "make_synthetic_datasets", return_value=[]), \
                 patch("sys.stderr", new_callable=io.StringIO):
                result = experiment.compare(args)
            self.assertEqual([request["strategy"] for request in requests],
                             ["greedy", "dp", "dp", "greedy", "greedy", "dp", "dp", "greedy", "greedy", "dp"])
            self.assertEqual({request["model_sha256"] for request in requests},
                             {experiment.digest(experiment.canonical_bytes(model))})
            self.assertTrue(all(request["model"] == model for request in requests))
            self.assertEqual(result["datasets"][0]["dp_token_savings"], 1)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
            self.assertEqual(result["datasets"][0]["results"]["greedy"]["totals"]["tokens"], 3)
            self.assertEqual(result["datasets"][0]["results"]["dp"]["totals"]["tokens"], 2)


if __name__ == "__main__":
    unittest.main()
