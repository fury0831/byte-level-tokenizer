"""Untimed record-trace selection and complete three-encoder output checks."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.inspect_cases import inspect_cases, main, select_requested_records


class CaseTraceTests(unittest.TestCase):
    def test_record_selection_preserves_request_order_and_rejects_ambiguity(self):
        first = {"id": "a", "category": "code", "text": "a"}
        second = {"id": "b", "category": "unicode", "text": "你"}
        self.assertEqual(select_requested_records([first, second], ["b", "a"]), [second, first])
        for records, requested in (([first], ["missing"]), ([first, first], ["a"]),
                                   ([first], ["a", "a"]), ([first], [])):
            with self.subTest(records=records, requested=requested), self.assertRaises(ValueError):
                select_requested_records(records, requested)

    def test_three_encoder_traces_roundtrip_and_refuse_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, evaluation_path, output = root / "model.json", root / "eval.jsonl", root / "traces.json"
            # BPE merges ab before bc, so ab|c|d; DP can still choose a|bcd.
            model = {"format": "msdm5051-tokenizer-v1", "tokens_hex":
                     [bytes([i]).hex() for i in range(256)] + [b"ab".hex(), b"bc".hex(), b"bcd".hex()],
                     "state": {"merges": [[97, 98, 256], [98, 99, 257], [257, 100, 258]]}}
            records = [{"id": "counterexample", "category": "code", "text": "abcd"},
                       {"id": "multibyte", "category": "unicode", "text": "你\n"}]
            model_path.write_text(json.dumps(model), encoding="utf-8")
            evaluation_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            original_model_bytes = model_path.read_bytes()
            result = inspect_cases(model_path, evaluation_path, ["multibyte", "counterexample"])
            self.assertEqual([r["id"] for r in result["records"]], ["multibyte", "counterexample"])
            strategies = result["records"][1]["strategies"]
            self.assertEqual(strategies["bpe"]["ids"], [256, 99, 100])
            self.assertEqual(strategies["greedy"]["ids"], [256, 99, 100])
            self.assertEqual(strategies["dp"]["ids"], [97, 258])
            for record in result["records"]:
                for trace in record["strategies"].values():
                    self.assertEqual(trace["token_count"], len(trace["tokens"]))
                    self.assertEqual(trace["ids"], [t["id"] for t in trace["tokens"]])
                    self.assertEqual(bytes.fromhex("".join(t["hex"] for t in trace["tokens"])),
                                     record["text"].encode("utf-8"))
            self.assertEqual(model_path.read_bytes(), original_model_bytes)
            args = ["--model", str(model_path), "--evaluation", str(evaluation_path),
                    "--record-id", "multibyte", "--record-id", "counterexample", "--output", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
            saved = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
