import unittest
from argparse import Namespace

from mistral_ruler_quality import METHODS, pending_methods, summarize


class RulerQualityTests(unittest.TestCase):
    def test_resume_with_existing_methods_only_leaves_global_topk(self):
        source_id = "qa_1:7"
        completed = {(source_id, method) for method in METHODS
                     if method != "global_empirical_topk_r20"}
        self.assertEqual(
            pending_methods(source_id, completed),
            ("global_empirical_topk_r20",),
        )

    def test_selected_methods_are_respected(self):
        source_id = "qa_1:7"
        self.assertEqual(
            pending_methods(source_id, set(), ("fullkv", "formula_waterfill_r20")),
            ("fullkv", "formula_waterfill_r20"),
        )

    def test_summary_rejects_incomplete_method(self):
        args = Namespace(tasks=["qa_1", "cwe"], num_samples=1, methods=list(METHODS))
        rows = [
            {"source_id": source_id, "method": method}
            for source_id in ("qa_1:1", "cwe:2")
            for method in METHODS
            if not (source_id == "cwe:2" and method == "global_empirical_topk_r20")
        ]
        with self.assertRaisesRegex(ValueError, "exactly 2 complete rows"):
            summarize(rows, args, lambda frame: {})

    def test_summary_supports_selected_method_without_fullkv(self):
        args = Namespace(
            tasks=["qa_1"], num_samples=1, methods=["formula_waterfill_r20"],
            model_path="model", context_length=32768, seed=42, target_kv_ratio=0.2,
            block_size=128, sink_tokens=4, local_tokens=512,
        )
        rows = [{
            "source_id": "qa_1:1", "method": "formula_waterfill_r20",
            "task": "qa_1", "answers": ["answer"], "prediction": "answer",
            "prompt_tokens": 100, "output_tokens": 1, "decode_seconds": 0.1,
        }]
        report = summarize(rows, args, lambda frame: {"qa_1": {"string_match": 100.0}})
        self.assertEqual(report["results"]["formula_waterfill_r20"]["macro_average"], 100.0)
        self.assertNotIn(
            "macro_retention_vs_fullkv", report["results"]["formula_waterfill_r20"]
        )


if __name__ == "__main__":
    unittest.main()
