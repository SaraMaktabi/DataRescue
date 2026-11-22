"""XGBoost avec les mêmes entrées, périodes et métriques que Random Forest."""

import argparse
import json
from pathlib import Path

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.prediction.train_random_forest import build_preprocessing, train


def build_xgboost(seed: int = 42, trees: int = 200,
                  scale_pos_weight: float = 1.0) -> Pipeline:
    return Pipeline([
        ("preprocessing", build_preprocessing()),
        ("classifier", XGBClassifier(
            n_estimators=trees, max_depth=3, learning_rate=0.05,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, tree_method="hist",
            objective="binary:logistic", eval_metric="logloss",
            random_state=seed, n_jobs=1,
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/prediction/history.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/prediction/xgboost"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        report = train(args.input, args.output_dir, args.seed, model_kind="xgboost")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for key in ("baseline", "xgboost_threshold_0_5", "xgboost_same_alert_count"):
        print(key, json.dumps(report[key]))
    print(f"Average precision: {report['average_precision']}")
    print(f"Rapport et modele : {args.output_dir}")
    print("Test reserve, non evalue.")


if __name__ == "__main__":
    main()
