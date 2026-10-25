"""Independent byte/model contract checks for the stage-2 foundation.

Run from src: python3 -B -m unittest discover -s tests -v
The fixtures and expected outputs below do not use the baseline encoder. Tests
with a byte-only vocabulary remain valid when the student encoder later gains
greedy or dynamic-programming segmentation.
"""

import copy
import json
import random
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import student_adapter as adapter


FORMAT = "msdm5051-tokenizer-v1"
CONTRACT_ERRORS = (ValueError, TypeError, UnicodeError)


def byte_model():
    """Construct the fixed alphabet independently of any training routine."""
    return {"format": FORMAT, "tokens_hex": [f"{value:02x}" for value in range(256)]}


def augmented_model():
    model = byte_model()
    # ID 256: AB; ID 257: the first two bytes of 你; ID 258: half of 😀.
    model["tokens_hex"].extend(["4142", "e4bd", "f09f"])
    model["state"] = {"fixture": "explicit byte-contract example"}
    return model


def canonical(model):
    return json.dumps(
        model, ensure_ascii=True, sort_keys=True, allow_nan=False, separators=(",", ":")
    )


class ByteContractTests(unittest.TestCase):
    def setUp(self):
        self.model = byte_model()
        self.runtime = adapter.prepare(self.model)

    def assert_byte_reconstruction(self, text, runtime=None, model=None):
        runtime = self.runtime if runtime is None else runtime
        model = self.model if model is None else model
        ids = adapter.encode(text, runtime)
        self.assertIsInstance(ids, list)
        self.assertTrue(all(type(token_id) is int for token_id in ids))
        self.assertTrue(all(0 <= token_id < len(model["tokens_hex"]) for token_id in ids))
        rebuilt = b"".join(bytes.fromhex(model["tokens_hex"][token_id]) for token_id in ids)
        self.assertEqual(rebuilt, text.encode("utf-8"))
        self.assertEqual(adapter.decode(ids, runtime), text)
        return ids

    def test_explicit_utf8_byte_ids(self):
        # Expected bytes are written explicitly, not obtained from encode().
        expected = [65, 195, 169, 228, 189, 160, 240, 159, 152, 128, 13, 10, 0]
        text = "Aé你😀\r\n\0"
        self.assertEqual(adapter.encode(text, self.runtime), expected)
        self.assertEqual(adapter.decode(expected, self.runtime), text)

    def test_all_fixed_ids_have_their_corresponding_byte(self):
        for value in range(256):
            with self.subTest(byte=value):
                # Check permitted high bytes inside valid UTF-8 sequences;
                # forbidden bytes C0/C1/F5-FF must instead raise an error.
                if value < 0x80:
                    ids = [value]
                elif value < 0xC0:
                    ids = [0xC2, value]
                elif value < 0xC2:
                    with self.assertRaises(CONTRACT_ERRORS):
                        adapter.decode([value], self.runtime)
                    continue
                elif value < 0xE0:
                    ids = [value, 0x80]
                elif value < 0xF0:
                    ids = [value, 0xA0 if value == 0xE0 else 0x80, 0x80]
                elif value <= 0xF4:
                    ids = [value, 0x90 if value == 0xF0 else 0x80, 0x80, 0x80]
                else:
                    with self.assertRaises(CONTRACT_ERRORS):
                        adapter.decode([value], self.runtime)
                    continue
                expected = bytes(ids).decode("utf-8", errors="strict")
                self.assertEqual(adapter.decode(ids, self.runtime), expected)
                self.assertEqual(adapter.encode(expected, self.runtime), ids)

    def test_empty_text_and_empty_ids(self):
        self.assertEqual(adapter.encode("", self.runtime), [])
        self.assertEqual(adapter.decode([], self.runtime), "")

    def test_whitespace_crlf_nul_and_repeated_bytes(self):
        for text in [" \t\n\r\n  ", "\0a\0", "a" * 257, "\n" * 31, "x = 1\r\n\tx += 1\n"]:
            with self.subTest(text=repr(text)):
                self.assert_byte_reconstruction(text)

    def test_normalization_forms_are_preserved_separately(self):
        precomposed = "é"
        combining = "e\u0301"
        self.assertEqual(self.assert_byte_reconstruction(precomposed), [0xC3, 0xA9])
        self.assertEqual(self.assert_byte_reconstruction(combining), [0x65, 0xCC, 0x81])
        self.assertNotEqual(adapter.encode(precomposed, self.runtime), adapter.encode(combining, self.runtime))

    def test_unseen_unicode_and_emoji_after_ascii_only_training(self):
        model = adapter.train(["hello hello", "code = 1\n"], 272, 5051)
        runtime = adapter.prepare(model)
        for text in ["你好", "🧑🏽‍💻", "𠮷", "العربية", "\u0378", "\U0010ffff"]:
            with self.subTest(text=repr(text)):
                self.assert_byte_reconstruction(text, runtime, model)

    def test_additional_tokens_decode_without_a_prior_encode(self):
        runtime = adapter.prepare(augmented_model())
        self.assertEqual(adapter.decode([256], runtime), "AB")
        self.assertEqual(adapter.decode([257, 0xA0], runtime), "你")
        self.assertEqual(adapter.decode([258, 0x98, 0x80], runtime), "😀")
        self.assertEqual(adapter.decode([256, 257, 0xA0, 32, 258, 0x98, 0x80], runtime), "AB你 😀")

    def test_utf8_is_decoded_after_concatenation(self):
        runtime = adapter.prepare(augmented_model())
        # Neither token 257 nor byte A0 is valid UTF-8 alone; together they are.
        with self.assertRaises(CONTRACT_ERRORS):
            adapter.decode([257], runtime)
        with self.assertRaises(CONTRACT_ERRORS):
            adapter.decode([0xA0], runtime)
        self.assertEqual(adapter.decode([257, 0xA0], runtime), "你")

    def test_invalid_token_ids_are_rejected(self):
        class IntSubclass(int):
            pass

        for bad in [-1, 256, 10**30, True, False, 1.0, 1.5, "1", None, IntSubclass(1)]:
            with self.subTest(invalid_id=repr(bad), kind=type(bad).__name__):
                with self.assertRaises(CONTRACT_ERRORS):
                    adapter.decode([bad], self.runtime)

    def test_malformed_utf8_is_rejected(self):
        for ids in [[0xFF], [0x80], [0xC3], [0xC0, 0x80], [0xED, 0xA0, 0x80], [0xF4, 0x90, 0x80, 0x80], [0xE4, 65, 0xA0]]:
            with self.subTest(ids=ids):
                with self.assertRaises(CONTRACT_ERRORS):
                    adapter.decode(ids, self.runtime)

    def test_record_order_and_repeat_calls_do_not_change_encoding(self):
        texts = ["hello", "你好", "", "\r\n", "hello", "e\u0301", "😀"]
        model = adapter.train(["hello hello", "你好你好"], 272, 5051)
        first_runtime = adapter.prepare(model)
        expected = {text: adapter.encode(text, first_runtime) for text in texts}
        reverse_runtime = adapter.prepare(model)
        for text in reversed(texts):
            self.assertEqual(adapter.encode(text, reverse_runtime), expected[text])
        for text in texts:
            self.assertEqual(adapter.encode(text, first_runtime), expected[text])

    def test_fixed_seed_generated_valid_unicode(self):
        rng = random.Random(5051)
        for case in range(64):
            # Map uniform scalar indices around the surrogate range.
            values = [rng.randrange(0x110000 - 0x800) for _ in range(rng.randrange(0, 49))]
            text = "".join(chr(value if value < 0xD800 else value + 0x800) for value in values)
            with self.subTest(case=case):
                self.assert_byte_reconstruction(text)


class ModelContractTests(unittest.TestCase):
    def test_prepare_and_calls_do_not_mutate_saved_model(self):
        model = augmented_model()
        before = canonical(model)
        runtime = adapter.prepare(model)
        adapter.encode("AB你 😀\r\n", runtime)
        adapter.decode([256, 257, 0xA0], runtime)
        with self.assertRaises(CONTRACT_ERRORS):
            adapter.decode([-1], runtime)
        self.assertEqual(canonical(model), before)

    def test_prepare_returns_fresh_independent_runtime(self):
        model = augmented_model()
        first = adapter.prepare(model)
        second = adapter.prepare(model)
        self.assertIsNot(first, second)
        expected = adapter.encode("AB你 😀", second)
        returned_ids = adapter.encode("AB你 😀", first)
        returned_ids.clear()
        self.assertEqual(adapter.encode("AB你 😀", first), expected)
        self.assertEqual(adapter.encode("AB你 😀", second), expected)
        # Runtime data must not alias the caller's mutable JSON token list.
        model["tokens_hex"][256] = "4344"
        self.assertEqual(adapter.decode([256], first), "AB")
        self.assertEqual(adapter.decode([256], second), "AB")

    def test_training_is_deterministic_and_preserves_input_records(self):
        records = ["abababab", "abab", "你好你好", "x = 1\n" * 3, ""]
        original = records.copy()
        first = adapter.train(records, 272, 5051)
        second = adapter.train(records, 272, 5051)
        self.assertEqual(canonical(first), canonical(second))
        self.assertEqual(records, original)
        self.assertEqual(first["format"], FORMAT)
        self.assertLessEqual(len(first["tokens_hex"]), 272)
        self.assertLessEqual(len(canonical(first).encode("utf-8")), 1024 * 1024)

    def test_learned_tokens_have_training_provenance(self):
        records = ["abababab", "你好你好", "x = 1\nx = 1\n"]
        model = adapter.train(records, 280, 5051)
        tokens = [bytes.fromhex(value) for value in model["tokens_hex"]]
        self.assertEqual(tokens[:256], [bytes([value]) for value in range(256)])
        self.assertEqual(len(tokens), len(set(tokens)))
        self.assertTrue(all(tokens))
        raw_records = [text.encode("utf-8") for text in records]
        for token in tokens[256:]:
            self.assertTrue(any(token in record for record in raw_records), token.hex())

    def test_training_does_not_merge_across_record_boundaries(self):
        records = ["a", "b", "a", "b"]
        model = adapter.train(records, 260, 5051)
        # No individual record has any multi-byte substring, so no learned
        # multi-byte token can be justified regardless of training algorithm.
        self.assertEqual(len(model["tokens_hex"]), 256)

    def test_trained_model_json_roundtrip(self):
        model = adapter.train(["abababab", "你好你好", "x = 1\n" * 2], 272, 5051)
        reloaded = json.loads(canonical(model))
        original_runtime = adapter.prepare(model)
        reloaded_runtime = adapter.prepare(reloaded)
        for text in ["abab", "你好", "unseen 🐱\r\n", ""]:
            ids = adapter.encode(text, original_runtime)
            self.assertEqual(adapter.encode(text, reloaded_runtime), ids)
            self.assertEqual(adapter.decode(ids, reloaded_runtime), text)

    def test_fresh_process_uses_only_saved_model_for_decode(self):
        script = r'''
import json
from pathlib import Path
import sys
import student_adapter as adapter

model = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
runtime = adapter.prepare(model)
# Direct decoding comes first: neither train nor encode has been called here.
decoded = adapter.decode([256, 257, 160, 32, 258, 152, 128], runtime)
ids = adapter.encode("unseen 🐱\r\n", runtime)
print(json.dumps({"direct_decode": decoded, "roundtrip": adapter.decode(ids, runtime)}))
'''
        with tempfile.TemporaryDirectory(prefix="tokenizer-contract-") as directory:
            path = Path(directory) / "model.json"
            path.write_text(canonical(augmented_model()), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-B", "-c", script, str(path)],
                cwd=SRC,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=15,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"direct_decode": "AB你 😀", "roundtrip": "unseen 🐱\r\n"})

    def test_malformed_model_structure_is_rejected(self):
        variants = [("not an object", []), ("missing tokens", {"format": FORMAT})]
        for description, mutation in [
            ("wrong format", lambda model: model.update(format="unknown")),
            ("too few fixed bytes", lambda model: model["tokens_hex"].pop()),
            ("incorrect fixed byte", lambda model: model["tokens_hex"].__setitem__(0, "01")),
            ("duplicate token", lambda model: model["tokens_hex"].append("00")),
            ("empty token", lambda model: model["tokens_hex"].append("")),
            ("invalid hex", lambda model: model["tokens_hex"].append("zz")),
            ("incomplete hex", lambda model: model["tokens_hex"].append("a")),
            ("non-string token", lambda model: model["tokens_hex"].append(42)),
        ]:
            model = byte_model()
            mutation(model)
            variants.append((description, model))
        for description, model in variants:
            with self.subTest(description=description):
                with self.assertRaises(CONTRACT_ERRORS):
                    adapter.prepare(model)

    def test_non_json_or_nonfinite_model_state_is_rejected(self):
        for state in [{"value": float("nan")}, {"value": float("inf")}, {"value": -float("inf")}, {"value": {1, 2}}]:
            model = byte_model()
            model["state"] = copy.deepcopy(state)
            with self.subTest(state=repr(state)):
                with self.assertRaises(CONTRACT_ERRORS):
                    adapter.prepare(model)


if __name__ == "__main__":
    unittest.main()
