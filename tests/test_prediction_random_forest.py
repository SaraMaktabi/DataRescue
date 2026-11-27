import csv
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.prediction.generate_history import FEATURE_COLUMNS, generate_history, write_history
from src.prediction.train_random_forest import build_model, feature_frame, top_k_alerts, train


class RandomForestTests(unittest.TestCase):
    def test_features_exclude_outcomes(self):
        frame = feature_frame(generate_history(2))
        self.assertEqual(list(frame.columns), list(FEATURE_COLUMNS))
        self.assertNotIn("failed", frame)
        self.assertNotIn("observed_duration_s", frame)

    def test_top_k_ties_and_limits(self):
        self.assertEqual(top_k_alerts([0.2, 0.8, 0.8, 0.1], 2), [0, 1, 1, 0])
        self.assertEqual(top_k_alerts([0.8, 0.8], 1), [1, 0])
        self.assertEqual(top_k_alerts([0.8], 0), [0])
        with self.assertRaises(ValueError):
            top_k_alerts([0.8], 2)

    def test_preprocessing_fitted_only_on_training_and_unknown_task(self):
        rows = generate_history(30)
        model = build_model(trees=5)
        model.fit(feature_frame(rows), [r["failed"] for r in rows])
        imputer = model.named_steps["preprocessing"].named_transformers_["numeric"]
        statistics = imputer.statistics_.copy()
        unseen = [rows[0] | {"task_id": "new_task", "input_rows": 99999999}]
        scores = model.predict_proba(feature_frame(unseen))
        np.testing.assert_array_equal(statistics, imputer.statistics_)
        self.assertEqual(scores.shape, (1, 2))
        self.assertTrue(np.isfinite(scores).all())

    def test_training_export_reload_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "history.csv"
            rows = generate_history(60)
            write_history(rows, source)
            first = train(source, root / "first", trees=10)
            second = train(source, root / "second", trees=10)
            self.assertEqual(first, second)
            self.assertFalse(first["test_evaluated"])
            self.assertEqual(first["splits"]["train"]["rows"], 420)
            self.assertEqual(first["baseline"]["alert_rate"],
                             first["random_forest_same_alert_count"]["alert_rate"])
            model = joblib.load(root / "first" / "model.joblib")
            self.assertEqual(model.predict_proba(feature_frame(rows[:2])).shape, (2, 2))
            with self.assertRaises(FileExistsError):
                train(source, root / "first", trees=10)

    def test_reject_single_class_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            write_history([r | {"failed": 0} for r in generate_history(20)], path)
            with self.assertRaises(ValueError):
                train(path, Path(directory) / "output", trees=5)

    def test_validation_threshold_is_applied_to_future_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "history.csv"
            output = root / "model"
            write_history(generate_history(60), source)
            report = train(source, output, trees=5, evaluate_test=True)
            point = report["operating_point"]
            self.assertLessEqual(point["alert_rate"], 0.15)
            self.assertTrue(report["test_evaluated"])
            with (output / "test_predictions.csv").open(encoding="utf-8", newline="") as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual(len(predictions), report["splits"]["test"]["rows"])
            for row in predictions:
                self.assertGreater(row["date"], report["splits"]["validation"]["end"])
                self.assertEqual(int(row["selected_alert"]),
                                 int(float(row["failure_score"]) >= point["threshold"]))
