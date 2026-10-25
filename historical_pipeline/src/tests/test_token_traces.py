"""Focused correctness checks for trace output, without timings or training."""

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

from experiments.inspect_tokens import escaped_text, main, render_report, select_records, table_code, trace_tokens
from tokenizer_core import build_runtime


def fixture_model(extras=()):
    return {"format": "msdm5051-tokenizer-v1", "tokens_hex":
            [bytes([i]).hex() for i in range(256)] + [token.hex() for token in extras]}


class TokenTraceTests(unittest.TestCase):
    def test_cross_character_utf8_tokens_have_exact_offsets_and_no_replacement(self):
        # 你 = e4 bd a0; 😀 = f0 9f 98 80. Token 257 crosses characters.
        runtime = build_runtime(fixture_model((bytes.fromhex("e4bd"),
                                               bytes.fromhex("a0f09f"),
                                               bytes.fromhex("9880"))))
        rows = trace_tokens("你😀", [256, 257, 258], runtime)
        self.assertEqual([(r["start"], r["end"]) for r in rows], [(0, 2), (2, 5), (5, 7)])
        self.assertEqual([r["hex"] for r in rows], ["e4bd", "a0f09f", "9880"])
        self.assertEqual([r["display"] for r in rows], ["partial UTF-8"] * 3)
        self.assertNotIn("�", str(rows))

    def test_ascii_whitespace_and_empty_spans(self):
        runtime = build_runtime(fixture_model((b"ab", b"\n\t")))
        rows = trace_tokens("ab\n\tc", [256, 257, 99], runtime)
        self.assertEqual([(r["index"], r["start"], r["end"], r["id"]) for r in rows],
                         [(0, 0, 2, 256), (1, 2, 4, 257), (2, 4, 5, 99)])
        self.assertEqual(rows[1]["display"], '"\\n\\t"')
        self.assertEqual(trace_tokens("", [], runtime), [])

    def test_invalid_ids_malformed_utf8_and_wrong_text_rejected(self):
        runtime = build_runtime(fixture_model())
        for ids in ([True], [-1], [256], [1.5]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                trace_tokens("a", ids, runtime)
        with self.assertRaises(TypeError):
            trace_tokens("a", (97,), runtime)
        with self.assertRaises(UnicodeDecodeError):
            trace_tokens("a", [255], runtime)
        with self.assertRaises(ValueError):
            trace_tokens("a", [98], runtime)

    def test_json_display_is_reversible_and_markdown_safe(self):
        text = '```\n|<script>&\x00\t"\\\r\u202e你'
        escaped = escaped_text(text)
        self.assertEqual(json.loads(escaped), text)
        self.assertNotIn("`", escaped)
        self.assertNotIn("\n", escaped)
        self.assertNotIn("\x00", escaped)
        self.assertIn("\\u202e", escaped)
        cell = table_code(escaped)
        self.assertNotIn("|", cell)
        self.assertNotIn("<script>", cell)
        self.assertIn("&#124;", cell)
        self.assertIn("&lt;script&gt;", cell)

    def test_selection_is_byte_shortest_and_order_stable(self):
        records = [{"id": "empty", "category": "english", "text": ""},
                   {"id": "three-bytes", "category": "english", "text": "你"},
                   {"id": "winner", "category": "english", "text": "aa"},
                   {"id": "later-tie", "category": "english", "text": "bb"}]
        records += [{"id": category, "category": category, "text": "a"}
                    for category in ("chinese", "code", "unicode")]
        selected = select_records(records)
        self.assertEqual(selected[0], ("english", 3, records[2]))
        with self.assertRaises(ValueError):
            select_records(records[:4])

    def test_report_full_examples_hashes_and_cli_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.json"
            evaluation = root / "eval.jsonl"
            output = root / "traces.md"
            model.write_text(json.dumps(fixture_model((b"ab", b"bcd"))), encoding="utf-8")
            texts = ("abcd", "你", "```|<x>\n\x00", "😀")
            records = [{"id": f"source-{i}", "category": category, "text": text}
                       for i, (category, text) in enumerate(zip(
                           ("english", "chinese", "code", "unicode"), texts))]
            evaluation.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            report = render_report(model, evaluation)
            self.assertIn("Canonical model SHA-256:", report)
            self.assertIn(str(model.resolve()), report)
            self.assertIn(str(evaluation.resolve()), report)
            self.assertIn("[256, 99, 100]", report)
            self.assertIn("[97, 257]", report)
            self.assertIn("E[97] -> E[257]", report)
            self.assertIn("partial UTF-8", report)
            for text in texts:
                self.assertIn(escaped_text(text), report)
            args = ["--model", str(model), "--evaluation", str(evaluation),
                    "--strategy", "dp", "--output", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            saved = output.read_bytes()
            self.assertNotIn("### greedy", saved.decode("utf-8"))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
