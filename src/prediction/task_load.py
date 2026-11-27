"""Mesure la surcharge par rapport au volume habituel de chaque tâche."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class TaskLoadFeatures(TransformerMixin, BaseEstimator):
    def fit(self, frame, y=None):
        self.typical_volume_ = frame.groupby("task_id")["input_rows"].median()
        return self

    def transform(self, frame):
        check_is_fitted(self)
        typical = frame["task_id"].map(self.typical_volume_)
        typical = typical.where(typical > 0, frame["volume_mean_7d"])
        return frame.assign(
            relative_load=frame["input_rows"].div(typical.where(typical > 0)),
            previous_memory_ratio=frame["previous_peak_memory_mb"] / frame["memory_limit_mb"],
        )[["input_null_rate_pct", "arrival_delay_min", "relative_load",
           "volume_delta_pct", "previous_memory_ratio"]]
