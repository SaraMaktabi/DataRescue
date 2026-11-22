import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.prediction.baseline import chronological_split
from src.prediction.generate_history import generate_history, write_history
from src.prediction.train_random_forest import feature_frame, train
from src.prediction.train_xgboost import build_xgboost


class XGBoostTests(unittest.TestCase):
    def test_unknown_task_and_missing_history(self):
        rows = generate_history(30)
        model = build_xgboost(trees=5)
        model.fit(feature_frame(rows), [r["failed"] for r in rows])
        imputer = model.named_steps["preprocessing"].named_transformers_["numeric"]
        before = imputer.statistics_.copy()
        scores = model.predict_proba(feature_frame([rows[0] | {"task_id": "unseen"}]))
        np.testing.assert_array_equal(before, imputer.statistics_)
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(scores.shape, (1, 2))

    def test_training_weight_reproducibility_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = generate_history(60)
            source = root / "input.csv"
            write_history(rows, source)
            report = train(source, root / "a", trees=10, model_kind="xgboost")
            again = train(source, root / "b", trees=10, model_kind="xgboost")
            self.assertEqual(report, again)
            training = chronological_split(rows)["train"]
            positives = sum(r["failed"] for r in training)
            self.assertEqual(report["model_parameters"]["scale_pos_weight"],
                             (len(training) - positives) / positives)
            self.assertFalse(report["test_evaluated"])
            self.assertEqual(report["baseline"]["alert_rate"],
                             report["xgboost_same_alert_count"]["alert_rate"])
            a = joblib.load(root / "a" / "model.joblib")
            b = joblib.load(root / "b" / "model.joblib")
            np.testing.assert_allclose(a.predict_proba(feature_frame(rows)),
                                       b.predict_proba(feature_frame(rows)))

    def test_test_outcomes_do_not_change_model_or_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = generate_history(60)
            cutoff = chronological_split(rows)["test"][0]["date"]
            altered = [r | {"failed": 1 - r["failed"]} if r["date"] >= cutoff else r for r in rows]
            write_history(rows, root / "first.csv")
            write_history(altered, root / "second.csv")
            first = train(root / "first.csv", root / "a", trees=5, model_kind="xgboost")
            second = train(root / "second.csv", root / "b", trees=5, model_kind="xgboost")
            first.pop("input_sha256")
            second.pop("input_sha256")
            self.assertEqual(first, second)
