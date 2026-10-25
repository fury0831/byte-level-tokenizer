"""Independent context-boundary tests for the result audit (not a new packer)."""

import unittest
from experiments.summarize_results import audit_context, cell, require_same_model


class ContextAuditTests(unittest.TestCase):
    def fixture(self, retained=1, tokens=100, byte_count=100, code_points=50):
        # Category list order deliberately differs from input file order.
        result = {"context_budget":{"budget_tokens":128,"records":retained,"tokens":tokens,"bytes":byte_count,"code_points":code_points},
                  "categories":[{"record_results":[{"id":"later","tokens":1}]},
                                {"record_results":[{"id":"first","tokens":100},{"id":"blocked","tokens":40}]}]}
        rows = [{"id":"first","category":"z","text":"é"*50}, {"id":"blocked","category":"z","text":"long"*10}, {"id":"later","category":"a","text":"x"}]
        return result, rows

    def test_stops_at_first_blocked_record_in_file_order(self):
        result, rows = self.fixture()
        audit = audit_context(result, rows)
        self.assertEqual(audit["included_ids"], ["first"])
        self.assertEqual(audit["first_excluded_id"], "blocked")
        self.assertEqual(audit["tokens_if_added"], 140)

    def test_rejects_skipping_a_record_or_early_stop(self):
        result, rows = self.fixture(tokens=101, byte_count=101, code_points=51)
        with self.assertRaises(ValueError):
            audit_context(result, rows)
        result, rows = self.fixture(retained=0,tokens=0,byte_count=0,code_points=0)
        with self.assertRaises(ValueError):
            audit_context(result, rows)

    def test_zero_retained_when_first_record_exceeds_budget(self):
        result, rows = self.fixture(retained=0,tokens=0,byte_count=0,code_points=0)
        result["categories"][1]["record_results"][0]["tokens"] = 129
        rows[0]["text"] = "x" * 129
        self.assertEqual(audit_context(result, rows)["tokens_if_added"],129)

    def test_exact_fit_then_empty_record_then_stop(self):
        rows=[{"id":"full","text":"A"*128},{"id":"empty","text":""},{"id":"next","text":"B"}]
        result={"context_budget":{"budget_tokens":128,"records":2,"tokens":128,"bytes":128,"code_points":128},
                "categories":[{"record_results":[{"id":"full","tokens":128},{"id":"empty","tokens":0},{"id":"next","tokens":1}]}]}
        self.assertEqual(audit_context(result,rows)["included_ids"],["full","empty"])

    def test_missing_fields_are_not_rendered_as_zero(self):
        self.assertEqual(cell(None),"not produced")
        self.assertEqual(cell(0),"0")

    def test_equal_counts_cannot_hide_different_model_identities(self):
        first = {"training":{"model_sha256":"model-a"}, "aggregate":{"tokens":5}}
        same = {"training":{"model_sha256":"model-a"}, "aggregate":{"tokens":5}}
        other = {"training":{"model_sha256":"model-b"}, "aggregate":{"tokens":5}}
        require_same_model(first, same)
        with self.assertRaisesRegex(ValueError, "different models"):
            require_same_model(first, other)


if __name__ == "__main__":
    unittest.main()
