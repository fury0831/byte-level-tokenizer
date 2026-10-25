"""Independent algorithm checks for stage 3; no benchmark or timing assertions.

Run from src: python3 -B -m unittest discover -s tests -v
Small-case expected segmentations come from explicit traces and exhaustive
full-vocabulary enumeration, never from the production trie or DP recurrence.
"""

import copy
import itertools
import json
from pathlib import Path
import sys
import unittest


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import adapter_dp
import adapter_greedy
import baseline_bpe
import student_adapter
from tokenizer_core import build_runtime, decode_strict, encode_dp, encode_greedy, iter_matches


FORMAT = "msdm5051-tokenizer-v1"
ADAPTERS = (adapter_greedy, adapter_dp, student_adapter)


def fixture_model(extra_tokens=()):
    """Build an explicit, valid ID table without invoking any learner."""
    tokens = [bytes([value]) for value in range(256)] + list(extra_tokens)
    assert len(tokens) == len(set(tokens)) and all(tokens)
    return {"format": FORMAT, "tokens_hex": [token.hex() for token in tokens]}


def token_bytes(model):
    return tuple(bytes.fromhex(value) for value in model["tokens_hex"])


def all_segmentations(data, tokens, start=0):
    """Enumerate every legal path by scanning tokens; only for tiny inputs.

    Deliberately no trie, memoization, or shortest-path/DP recurrence. This
    oracle is structurally different from the production implementation.
    """
    if start == len(data):
        yield ()
        return
    for token_id, token in enumerate(tokens):
        if data.startswith(token, start):
            for suffix in all_segmentations(data, tokens, start + len(token)):
                yield (token_id,) + suffix


def exact_oracle(data, tokens):
    """Minimize count, then prefer a longer token at the first difference."""
    return list(min(
        all_segmentations(data, tokens),
        key=lambda ids: (len(ids), tuple((-len(tokens[i]), i) for i in ids)),
    ))


def greedy_scan_oracle(data, tokens):
    """Simple independent longest match: inspect the whole ID table."""
    result = []
    start = 0
    while start < len(data):
        choices = [i for i, token in enumerate(tokens) if data.startswith(token, start)]
        chosen = min(choices, key=lambda i: (-len(tokens[i]), i))
        result.append(chosen)
        start += len(tokens[chosen])
    return result


def binary_texts(max_length):
    for length in range(max_length + 1):
        for letters in itertools.product("ab", repeat=length):
            yield "".join(letters)


class BPETrainingTests(unittest.TestCase):
    def assert_learned(self, records, budget, learned, merges):
        expected_tokens = [f"{value:02x}" for value in range(256)]
        expected_tokens += [value.hex() for value in learned]
        for adapter in ADAPTERS:
            with self.subTest(adapter=adapter.__name__):
                model = adapter.train(records, budget, 5051)
                self.assertEqual(model["format"], FORMAT)
                self.assertEqual(model["tokens_hex"], expected_tokens)
                self.assertEqual(model["state"]["merges"], merges)

    def test_duplicate_records_contribute_their_frequency(self):
        # zz wins 3:2. Counting each distinct record once would tie and pick ab.
        self.assert_learned(["zz"] * 3 + ["ab"] * 2, 257,
                            [b"zz"], [[122, 122, 256]])

    def test_equal_pair_counts_use_numeric_pair_id_order(self):
        # ab wins the first 2:2:2 tie. Afterwards (z,a) beats (256,z),
        # although token bytes b'abz' would sort before b'za'.
        self.assert_learned(["abz", "abz", "za", "za"], 258,
                            [b"ab", b"za"],
                            [[97, 98, 256], [122, 97, 257]])

    def test_nonoverlapping_merge_trace(self):
        # Five a bytes become [aa, aa, a], never overlapping copies of aa.
        self.assertEqual(baseline_bpe.merge((97,) * 5, (97, 97), 256),
                         (256, 256, 97))
        self.assert_learned(["aaaaa", "aaaaa"], 259,
                            [b"aa", b"aaa", b"aaaaa"],
                            [[97, 97, 256], [256, 97, 257], [256, 257, 258]])

    def test_pair_frequency_counts_overlaps_before_nonoverlapping_merge(self):
        # aa has two adjacent occurrences in aaa, despite one actual merge.
        self.assert_learned(["aaa"], 260, [b"aa"], [[97, 97, 256]])

    def test_stops_below_two_occurrences_and_on_empty_data(self):
        for records in [[], [""], ["abc"], ["a", "b", "a", "b"]]:
            with self.subTest(records=records):
                self.assert_learned(records, 512, [], [])

    def test_budget_is_a_maximum_not_a_required_size(self):
        self.assert_learned(["abababab", "abababab"], 256, [], [])
        self.assert_learned(["abababab", "abababab"], 257,
                            [b"ab"], [[97, 98, 256]])
        self.assert_learned(["ab", "ab"], 512,
                            [b"ab"], [[97, 98, 256]])
        for adapter in ADAPTERS:
            with self.subTest(adapter=adapter.__name__):
                with self.assertRaises(ValueError):
                    adapter.train(["abab"], 255, 5051)

    def test_training_never_creates_cross_record_tokens(self):
        # Joining these records would invent repeated "ba" boundaries.
        self.assert_learned(["ab", "ab", "ab"], 512,
                            [b"ab"], [[97, 98, 256]])

    def test_adapters_share_the_exact_baseline_model(self):
        records = ["abababab", "你好你好", "x = 1\n" * 3, "", "abababab"]
        original = records.copy()
        for budget in [256, 257, 272, 512]:
            expected = baseline_bpe.train(records, budget, 5051)
            for adapter in ADAPTERS:
                with self.subTest(adapter=adapter.__name__, budget=budget):
                    first = adapter.train(records, budget, 5051)
                    second = adapter.train(records, budget, 5051)
                    self.assertEqual(first, expected)
                    self.assertEqual(second, expected)
                    self.assertLessEqual(len(first["tokens_hex"]), budget)
        self.assertEqual(records, original)


class TrieMatchingTests(unittest.TestCase):
    def test_every_terminal_is_returned_including_prefixes(self):
        runtime = build_runtime(fixture_model([b"ab", b"abc", b"abcd"]))
        self.assertEqual(sorted(iter_matches(b"abcdx", 0, runtime)),
                         [(1, 97), (2, 256), (3, 257), (4, 258)])
        self.assertEqual(sorted(iter_matches(b"zabcdx", 1, runtime)),
                         [(2, 97), (3, 256), (4, 257), (5, 258)])

    def test_unfinished_prefix_falls_back_to_last_terminal(self):
        runtime = build_runtime(fixture_model([b"ab", b"abcd"]))
        self.assertEqual(sorted(iter_matches(b"abcx", 0, runtime)),
                         [(1, 97), (2, 256)])
        self.assertEqual(encode_greedy("abcx", runtime), [256, 99, 120])
        self.assertEqual(encode_dp("abcx", runtime), [256, 99, 120])

    def test_zero_id_is_a_valid_terminal_not_a_false_sentinel(self):
        runtime = build_runtime(fixture_model([b"\x00a"]))
        self.assertEqual(sorted(iter_matches(b"\x00a", 0, runtime)),
                         [(1, 0), (2, 256)])
        for encoder in [encode_greedy, encode_dp]:
            self.assertEqual(encoder("\x00b", runtime), [0, 98])
            self.assertEqual(encoder("\x00a", runtime), [256])

    def test_trie_matches_raw_partial_utf8_bytes(self):
        runtime = build_runtime(fixture_model([b"\xe4\xbd", b"\xf0\x9f"]))
        self.assertEqual(sorted(iter_matches(b"\xe4\xbd\xa0", 0, runtime)),
                         [(1, 0xE4), (2, 256)])
        self.assertEqual(sorted(iter_matches(b"\xe4\xbd\xa0", 1, runtime)),
                         [(2, 0xBD)])
        for encoder in [encode_greedy, encode_dp]:
            ids = encoder("你😀", runtime)
            self.assertEqual(ids, [256, 0xA0, 257, 0x98, 0x80])
            self.assertEqual(decode_strict(ids, runtime), "你😀")

    def test_matches_agree_with_full_table_scan_at_every_byte_position(self):
        model = fixture_model([b"ab", b"aba", b"ba", b"\xe4\xbd", b"\xbd\xa0", b"\x00a"])
        tokens = token_bytes(model)
        runtime = build_runtime(model)
        data = "ababa你\x00a".encode("utf-8")
        for start in range(len(data)):
            expected = [(start + len(token), i) for i, token in enumerate(tokens)
                        if data.startswith(token, start)]
            with self.subTest(start=start):
                self.assertEqual(sorted(iter_matches(data, start, runtime)), sorted(expected))


class SegmentationTests(unittest.TestCase):
    def test_longest_match_can_lose_to_minimum_token_segmentation(self):
        runtime = build_runtime(fixture_model([b"ab", b"bcd"]))
        self.assertEqual(encode_greedy("abcd", runtime), [256, 99, 100])
        self.assertEqual(encode_dp("abcd", runtime), [97, 257])

    def test_equal_token_count_prefers_longer_current_token(self):
        runtime = build_runtime(fixture_model([b"ab", b"bc"]))
        # ab/c and a/bc are both two tokens; deterministic tie rule picks ab/c.
        self.assertEqual(encode_dp("abc", runtime), [256, 99])
        self.assertEqual(encode_dp("abcabc", runtime), [256, 99, 256, 99])

    def test_dp_matches_independent_exhaustive_oracle(self):
        vocabularies = [
            [b"ab", b"ba"],
            [b"aa", b"aaa", b"ab", b"bba"],
            [b"ab", b"aba", b"bab", b"bb", b"aab"],
            [b"aab", b"abb", b"baaa", b"bbbb"],
        ]
        for extra_tokens in vocabularies:
            model = fixture_model(extra_tokens)
            tokens = token_bytes(model)
            runtime = build_runtime(model)
            for text in binary_texts(7):
                with self.subTest(vocabulary=extra_tokens, text=text):
                    expected = exact_oracle(text.encode("ascii"), tokens)
                    actual = encode_dp(text, runtime)
                    self.assertEqual(actual, expected)
                    self.assertEqual(decode_strict(actual, runtime), text)

    def test_greedy_matches_independent_full_vocabulary_scan(self):
        model = fixture_model([b"aa", b"ab", b"aba", b"bba", b"bbab", b"\xe4\xbd"])
        tokens = token_bytes(model)
        runtime = build_runtime(model)
        texts = list(binary_texts(7)) + ["你abab", "ab\x00b", "ab\r\n", "a你b"]
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(encode_greedy(text, runtime),
                                 greedy_scan_oracle(text.encode("utf-8"), tokens))
        self.assertIn(258, encode_greedy("aba", runtime))

    def test_empty_and_long_records_need_no_recursive_segmentation(self):
        runtime = build_runtime(fixture_model([b"ab", b"abab"]))
        text = "abab" * 4096
        for encoder in [encode_greedy, encode_dp]:
            with self.subTest(encoder=encoder.__name__):
                self.assertEqual(encoder("", runtime), [])
                ids = encoder(text, runtime)
                self.assertEqual(ids, [257] * 4096)
                self.assertEqual(decode_strict(ids, runtime), text)

    def test_tokens_may_cross_unicode_codepoint_boundaries(self):
        # The middle token joins the final byte of 你 to the first two of 😀.
        model = fixture_model([b"\xe4\xbd", b"\xa0\xf0\x9f", b"\x98\x80"])
        runtime = build_runtime(model)
        for encoder in [encode_greedy, encode_dp]:
            self.assertEqual(encoder("你😀", runtime), [256, 257, 258])
        self.assertEqual(decode_strict([256, 257, 258], runtime), "你😀")
        for token_id in [256, 257, 258]:
            with self.assertRaises(UnicodeDecodeError):
                decode_strict([token_id], runtime)

    def test_prepared_state_and_model_are_not_changed_by_either_encoder(self):
        model = fixture_model([b"ab", b"bcd", b"\xe4\xbd"])
        before = copy.deepcopy(model)
        first, second = build_runtime(model), build_runtime(model)
        self.assertIsNot(first, second)
        texts = ["abcd", "你ab", "ab\r\n", "", "abcd"]
        expected = {encoder: {text: encoder(text, first) for text in texts}
                    for encoder in [encode_greedy, encode_dp]}
        for encoder in [encode_greedy, encode_dp]:
            for text in reversed(texts):
                self.assertEqual(encoder(text, second), expected[encoder][text])
            for text in texts:
                ids = encoder(text, first)
                self.assertEqual(ids, expected[encoder][text])
                ids.clear()
                self.assertEqual(encoder(text, first), expected[encoder][text])
        self.assertEqual(model, before)
        # Mutating the caller-owned JSON list cannot alter an existing runtime.
        model["tokens_hex"][256] = b"xy".hex()
        for runtime in [first, second]:
            self.assertEqual(encode_greedy("abcd", runtime), [256, 99, 100])
            self.assertEqual(encode_dp("abcd", runtime), [97, 257])
            self.assertEqual(decode_strict([256], runtime), "ab")

    def test_comparator_adapters_select_their_named_policy(self):
        model = fixture_model([b"ab", b"bcd"])
        for adapter, expected in [(adapter_greedy, [256, 99, 100]),
                                  (adapter_dp, [97, 257]),
                                  (student_adapter, [97, 257])]:
            with self.subTest(adapter=adapter.__name__):
                runtime = adapter.prepare(json.loads(json.dumps(model)))
                ids = adapter.encode("abcd", runtime)
                self.assertEqual(ids, expected)
                self.assertEqual(adapter.decode(ids, runtime), "abcd")


if __name__ == "__main__":
    unittest.main()
