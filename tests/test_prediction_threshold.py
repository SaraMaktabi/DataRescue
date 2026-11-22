import json
import tempfile
import unittest
from pathlib import Path

from src.prediction.generate_history import write_history
from src.prediction.tune_threshold import compare_thresholds, tune


class ThresholdTests(unittest.TestCase):
    def test_budget_recall_and_tie_break(self):
        _, selected = compare_thresholds([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6], 0.75)
        self.assertEqual(selected["threshold"], 0.7)
        self.assertEqual(selected["recall"], 1)
        _, selected = compare_thresholds([1, 0, 0, 0], [0.9, 0.8, 0.7, 0.6], 0.75)
        self.assertEqual(selected["alert_count"], 1)

    def test_tied_scores_cannot_be_split_to_fill_budget(self):
        _, selected = compare_thresholds([1, 1, 0, 0], [0.8, 0.8, 0.2, 0.1], 0.25)
        self.assertEqual(selected["alert_count"], 0)
        _, selected = compare_thresholds([1, 0], [1.0, 1.0], 0)
        self.assertGreater(selected["threshold"], 1)

    def test_invalid_inputs(self):
        for labels, scores, budget in [([], [], 0.15), ([0], [0.5], 0.15),
                                      ([1], [float("nan")], 0.15),
                                      ([1], [0.5], -0.1), ([1], [], 0.15)]:
            with self.assertRaises(ValueError):
                compare_thresholds(labels, scores, budget)

    def test_export_and_reject_test_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            metadata = {"evaluation_set": "validation", "test_evaluated": False,
                        "input_sha256": "example", "model_kind": "xgboost",
                        "splits": {"validation": {"rows": 2, "start": "2025-01-01", "end": "2025-01-02"}}}
            (model / "report.json").write_text(json.dumps(metadata), encoding="utf-8")
            (model / "model.joblib").write_bytes(b"hash-only-test-fixture")
            write_history([{"run_id": "a", "date": "2025-01-01", "failed": 1, "failure_score": 0.8},
                           {"run_id": "b", "date": "2025-01-02", "failed": 0, "failure_score": 0.2}],
                          model / "validation_predictions.csv")
            report = tune(model, root / "out", 0.5)
            self.assertEqual(report["selected"]["recall"], 1)
            self.assertFalse(report["test_evaluated"])
            self.assertTrue((root / "out" / "threshold_comparison.csv").exists())
            with self.assertRaises(FileExistsError):
                tune(model, root / "out", 0.5)
            metadata["evaluation_set"] = "test"
            (model / "report.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                tune(model, root / "rejected", 0.5)
