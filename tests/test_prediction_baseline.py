"""Tests du découpage, des seuils et des métriques de référence."""

import json
import tempfile
import unittest
from pathlib import Path

from src.prediction.baseline import (
    chronological_split, classification_metrics, load_history, predict, run,
)
from src.prediction.generate_history import generate_history, write_history


class BaselineTests(unittest.TestCase):
    def test_split_dates_disjoint_sorted_and_complete(self):
        rows = generate_history(20)
        splits = chronological_split(list(reversed(rows)))
        self.assertEqual([len(v) for v in splits.values()], [140, 30, 30])
        self.assertEqual(splits, chronological_split(rows))
        self.assertLess(splits["train"][-1]["date"], splits["validation"][0]["date"])
        self.assertLess(splits["validation"][-1]["date"], splits["test"][0]["date"])
        ids = [r["run_id"] for part in splits.values() for r in part]
        self.assertEqual(len(set(ids)), len(rows))
        self.assertEqual(set(ids), {r["run_id"] for r in rows})

    def test_invalid_split(self):
        for rows, ratios in [(generate_history(2), (0.7, 0.15)),
                             (generate_history(20), (0.8, 0.3)),
                             ([], (0.7, 0.15))]:
            with self.assertRaises(ValueError):
                chronological_split(rows, *ratios)

    def test_each_threshold_and_missing_history(self):
        normal = {"input_null_rate_pct": 1, "arrival_delay_min": 2,
                  "volume_delta_pct": None, "previous_peak_memory_mb": None,
                  "memory_limit_mb": 1000}
        self.assertEqual(predict(normal), {"alert": 0, "reasons": []})
        for column, value, reason in [
            ("input_null_rate_pct", 5, "high_null_rate"),
            ("arrival_delay_min", 30, "late_source"),
            ("volume_delta_pct", -50, "unusual_volume"),
            ("previous_peak_memory_mb", 900, "previous_memory_pressure"),
        ]:
            self.assertEqual(predict(normal | {column: value}),
                             {"alert": 1, "reasons": [reason]})
        self.assertEqual(predict(normal | {"failed": 1, "observed_peak_memory_mb": 99999}),
                         predict(normal))

    def test_metrics_known_confusion_matrix(self):
        metrics = classification_metrics([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["true_negatives"], 2)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["false_positive_rate"], 1 / 3)
        self.assertEqual(metrics["alert_rate"], 0.4)
        empty_positive = classification_metrics([0, 0], [0, 0])
        self.assertIsNone(empty_positive["precision"])
        self.assertIsNone(empty_positive["recall"])
        with self.assertRaises(ValueError):
            classification_metrics([1], [])

    def test_loader_validation_and_outcome_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            rows = generate_history(10)
            write_history(rows, path)
            loaded = load_history(path)
            self.assertIsNone(loaded[0]["previous_peak_memory_mb"])
            self.assertNotIn("simulated_cause", loaded[0])
            self.assertNotIn("observed_duration_s", loaded[0])
            for index, change in enumerate([
                {"failed": 2}, {"memory_limit_mb": 0},
                {"input_null_rate_pct": float("nan")}, {"date": "invalid"},
                {"arrival_delay_min": ""},
            ]):
                invalid = Path(directory) / f"invalid{index}.csv"
                write_history([rows[0] | change], invalid)
                with self.assertRaises(ValueError):
                    load_history(invalid)
            duplicate = Path(directory) / "duplicate.csv"
            write_history([rows[0], rows[0]], duplicate)
            with self.assertRaises(ValueError):
                load_history(duplicate)

    def test_exports_validation_only_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            output = Path(directory) / "baseline"
            write_history(generate_history(20), source)
            report = run(source, output)
            self.assertFalse(report["test_evaluated"])
            self.assertEqual(report["baseline"]["rows"], 30)
            self.assertEqual(json.loads((output / "report.json").read_text(encoding="utf-8")), report)
            self.assertEqual(len(load_history(output / "test.csv")), 30)
            self.assertFalse((output / "test_predictions.csv").exists())
            with self.assertRaises(FileExistsError):
                run(source, output)
