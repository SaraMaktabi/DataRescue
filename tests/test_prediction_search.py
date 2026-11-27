import unittest

from src.prediction.generate_history import generate_history
from src.prediction.search_xgboost import class_weight, select_parameters, temporal_folds
from src.prediction.task_load import TaskLoadFeatures
from src.prediction.train_random_forest import feature_frame


class SearchTests(unittest.TestCase):
    def test_task_load_uses_fitted_volume_and_falls_back_for_new_tasks(self):
        rows = generate_history(10)
        training = feature_frame(rows[:10])
        row = rows[0]
        features = TaskLoadFeatures().fit(training)
        future = feature_frame([
            row | {"input_rows": 2 * row["input_rows"]},
            row | {"task_id": "new_task", "input_rows": 300, "volume_mean_7d": 200},
        ])
        self.assertEqual(features.transform(future)["relative_load"].tolist(), [2.0, 1.5])

    def test_folds_respect_days_and_time(self):
        rows = generate_history(60)
        for past, future in temporal_folds(rows):
            self.assertLess(max(rows[i]["date"] for i in past),
                            min(rows[i]["date"] for i in future))
            self.assertFalse(set(past) & set(future))
            self.assertEqual(len(past) % 10, 0)
            self.assertEqual(len(future) % 10, 0)

    def test_weights_and_invalid_training(self):
        self.assertEqual(class_weight([0, 0, 0, 0, 1], "ratio"), 4)
        self.assertEqual(class_weight([0, 0, 0, 0, 1], "sqrt"), 2)
        self.assertEqual(class_weight([0, 1], "none"), 1)
        with self.assertRaises(ValueError):
            class_weight([0, 0], "ratio")

    def test_selection_reproducible_and_weights_per_fold(self):
        rows = generate_history(60)
        configurations = [dict(n_estimators=5, max_depth=2, weight_mode=mode)
                          for mode in ("ratio", "none")]
        result = select_parameters(rows, configurations=configurations)
        self.assertEqual(result, select_parameters(rows, configurations=configurations))
        self.assertEqual(result["best"]["mean_average_precision"],
                         max(r["mean_average_precision"] for r in result["candidates"]))
        for index, (past, _) in enumerate(temporal_folds(rows)):
            expected = class_weight([rows[i]["failed"] for i in past], "ratio")
            self.assertEqual(result["candidates"][0]["fold_class_weights"][index], expected)

    def test_reject_too_short_history(self):
        with self.assertRaises(ValueError):
            temporal_folds(generate_history(3))
