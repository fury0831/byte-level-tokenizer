"""Selected original byte/model checks using explicit fixtures only.

Training, the course checker and fresh-process course-adapter tests are omitted.
See release-validation.json for the latest recorded verification.
"""
import copy
import json
import random
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tokenizer_core import build_runtime, decode_strict, encode_dp

adapter = SimpleNamespace(prepare=build_runtime, encode=encode_dp, decode=decode_strict)
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
