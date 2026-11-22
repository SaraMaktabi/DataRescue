"""Contrats du dataset et absence de fuite temporelle."""

import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.prediction.generate_history import (
    FEATURE_COLUMNS, OUTCOME_COLUMNS, TASKS, generate_history, write_history,
)


class HistoryTests(unittest.TestCase):
    def test_reproducibility_and_prefix_stability(self):
        short = generate_history(10)
        self.assertEqual(short, generate_history(10))
        self.assertEqual(short, generate_history(20)[:len(short)])
        self.assertNotEqual(short, generate_history(10, seed=43))

    def test_shape_and_labels(self):
        rows = generate_history()
        self.assertEqual(len(rows), 365 * len(TASKS))
        self.assertEqual(len({r["run_id"] for r in rows}), len(rows))
        self.assertEqual({r["failed"] for r in rows}, {0, 1})
        self.assertFalse(set(FEATURE_COLUMNS) & set(OUTCOME_COLUMNS))
        for row in rows:
            self.assertTrue(set(FEATURE_COLUMNS).issubset(row))
            self.assertGreater(row["input_rows"], 0)
            self.assertTrue(0 <= row["input_null_rate_pct"] <= 100)
            self.assertEqual(row["simulated_cause"] == "none", row["failed"] == 0)

    def test_lags_and_windows_use_only_same_task_past(self):
        rows = generate_history(12)
        for task_id, *_ in TASKS:
            history = [r for r in rows if r["task_id"] == task_id]
            self.assertIsNone(history[0]["previous_duration_s"])
            self.assertIsNone(history[0]["volume_mean_7d"])
            self.assertIsNone(history[0]["duration_trend_7d_s"])
            for i in range(1, len(history)):
                row, previous = history[i], history[i - 1]
                self.assertEqual(row["history_count"], i)
                for feature, outcome in (
                    ("previous_duration_s", "observed_duration_s"),
                    ("previous_peak_memory_mb", "observed_peak_memory_mb"),
                    ("previous_retry_count", "observed_retry_count"),
                    ("previous_failed", "failed"),
                ):
                    self.assertEqual(row[feature], previous[outcome])
                window = history[max(0, i - 7):i]
                mean = sum(r["input_rows"] for r in window) / len(window)
                self.assertEqual(row["volume_mean_7d"], round(mean, 2))
                if len(window) > 1:
                    trend = (window[-1]["observed_duration_s"] - window[0]["observed_duration_s"]) / (len(window) - 1)
                    self.assertEqual(row["duration_trend_7d_s"], round(trend, 2))

    def test_calendar_and_invalid_days(self):
        rows = generate_history(7, start_date=date(2024, 2, 24))
        self.assertEqual({r["date"] for r in rows if r["is_month_end"]}, {"2024-02-29"})
        self.assertEqual({r["date"] for r in rows if r["is_payday"]}, {"2024-02-25"})
        with self.assertRaises(ValueError):
            generate_history(0)

    def test_csv_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            write_history(generate_history(1), path)
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 10)
            self.assertEqual(rows[0]["previous_duration_s"], "")
            with self.assertRaises(FileExistsError):
                write_history(generate_history(2), path)


if __name__ == "__main__":
    unittest.main()
