"""Entraînement et évaluation chronologique des modèles de panne."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.prediction.baseline import (
    chronological_split, classification_metrics, load_history, predict,
)
from src.prediction.generate_history import FEATURE_COLUMNS, write_history
from src.prediction.tune_threshold import compare_thresholds


def feature_frame(rows: list[dict]) -> pd.DataFrame:
    """Exclut explicitement cible, résultats et identifiants d'exécution."""
    frame = pd.DataFrame(rows).loc[:, list(FEATURE_COLUMNS)].copy()
    for column in FEATURE_COLUMNS:
        if column != "task_id":
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    return frame


def build_preprocessing() -> ColumnTransformer:
    numeric = [column for column in FEATURE_COLUMNS if column != "task_id"]
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True,
                                  keep_empty_features=True), numeric),
        ("task", OneHotEncoder(handle_unknown="ignore"), ["task_id"]),
    ])


def build_model(seed: int = 42, trees: int = 200) -> Pipeline:
    return Pipeline([
        ("preprocessing", build_preprocessing()),
        ("classifier", RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=5, class_weight="balanced",
            random_state=seed, n_jobs=1,
        )),
    ])


def top_k_alerts(scores: list[float], budget: int) -> list[int]:
    """Classement rétrospectif ; ex aequo départagés dans l'ordre des lignes."""
    if not 0 <= budget <= len(scores):
        raise ValueError("Budget hors limites")
    selected = set(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:budget])
    return [int(i in selected) for i in range(len(scores))]


def failure_scores(model: Pipeline, rows: list[dict]) -> list[float]:
    frame = feature_frame(rows)
    if isinstance(model.named_steps["classifier"], IsolationForest):
        # Une décision négative indique une anomalie. La transformation
        # conserve le classement et donne des scores compatibles avec le seuil.
        return expit(-model.decision_function(frame)).tolist()
    return model.predict_proba(frame)[:, 1].tolist()


def train(input_path: Path, output_dir: Path, seed: int = 42,
          trees: int = 200, *, model_kind: str = "random_forest",
          xgb_parameters: dict | None = None, max_alert_rate: float | None = None,
          evaluate_test: bool = False, task_load: bool = False) -> dict:
    if model_kind not in {"random_forest", "xgboost", "isolation_forest"}:
        raise ValueError("Modèle inconnu")
    if evaluate_test and max_alert_rate is None:
        max_alert_rate = 0.15
    if output_dir.exists():
        raise FileExistsError(f"Dossier existant : {output_dir}")
    splits = chronological_split(load_history(input_path))
    training, validation = splits["train"], splits["validation"]
    y_train = [r["failed"] for r in training]
    if model_kind != "isolation_forest" and set(y_train) != {0, 1}:
        raise ValueError("L'entraînement nécessite des succès et des pannes")
    versions = {}
    if model_kind == "xgboost":
        import xgboost
        from src.prediction.train_xgboost import build_xgboost

        weight = y_train.count(0) / y_train.count(1)
        model = build_xgboost(seed, trees, weight, task_load=task_load)
        parameters = {"n_estimators": trees, "max_depth": 3, "learning_rate": 0.05,
                      "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
                      "scale_pos_weight": weight, "tree_method": "hist",
                      "objective": "binary:logistic", "eval_metric": "logloss"}
        if xgb_parameters:
            parameters.update(xgb_parameters)
            model.named_steps["classifier"].set_params(**xgb_parameters)
        versions["xgboost_version"] = xgboost.__version__
    elif model_kind == "isolation_forest":
        parameters = {"n_estimators": trees, "contamination": "auto"}
        model = Pipeline([
            ("preprocessing", build_preprocessing()),
            ("classifier", IsolationForest(**parameters, random_state=seed, n_jobs=1)),
        ])
    else:
        model = build_model(seed, trees)
        parameters = {"n_estimators": trees, "min_samples_leaf": 5,
                      "class_weight": "balanced"}
    training_frame = feature_frame(training)
    if model_kind == "isolation_forest":
        model.fit(training_frame)
    else:
        model.fit(training_frame, y_train)
    scores = failure_scores(model, validation)
    labels = [r["failed"] for r in validation]
    fixed_alerts = [predict(row)["alert"] for row in validation]
    rf_alerts = [int(score >= 0.5) for score in scores]
    budget = sum(fixed_alerts)
    ranked_alerts = top_k_alerts(scores, budget)
    report = {
        "evaluation_set": "validation", "test_evaluated": False,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "seed": seed, "sklearn_version": sklearn.__version__,
        "model_kind": model_kind, **versions,
        "features": list(FEATURE_COLUMNS),
        "feature_profile": "task_load" if task_load else "standard",
        "score_kind": "anomaly_score" if model_kind == "isolation_forest" else "uncalibrated_probability",
        "model_parameters": parameters,
        "splits": {name: {"rows": len(rows), "start": rows[0]["date"],
                          "end": rows[-1]["date"]} for name, rows in splits.items()},
        "baseline": classification_metrics(labels, fixed_alerts),
        f"{model_kind}_threshold_0_5": classification_metrics(labels, rf_alerts),
        f"{model_kind}_same_alert_count": classification_metrics(labels, ranked_alerts),
        "comparison_alert_budget": budget,
        "average_precision": float(average_precision_score(labels, scores)) if any(labels) else None,
        "failure_prevalence": sum(labels) / len(labels),
        "notes": [
            "Average precision : résumé non interpolé de la courbe précision-rappel, pas aire trapézoïdale.",
            "Comparaison top-k rétrospective au nombre d'alertes de la baseline ; pas une politique en ligne.",
            "Seuil 0.5 fixé sans optimisation.",
        ],
    }
    test_predictions = []
    if max_alert_rate is not None:
        _, selected = compare_thresholds(labels, scores, max_alert_rate)
        threshold = selected["threshold"]
        report["operating_point"] = selected
        report["max_validation_alert_rate"] = max_alert_rate
        report["decision_rule"] = "failure_score >= operating_point.threshold"
        if evaluate_test:
            test = splits["test"]
            test_scores = failure_scores(model, test)
            test_labels = [row["failed"] for row in test]
            test_baseline = [predict(row)["alert"] for row in test]
            test_alerts = [int(score >= threshold) for score in test_scores]
            report["test_evaluated"] = True
            report["test"] = {
                "selected_threshold": classification_metrics(test_labels, test_alerts),
                "threshold_0_5": classification_metrics(
                    test_labels, [int(score >= 0.5) for score in test_scores]),
                "baseline": classification_metrics(test_labels, test_baseline),
                "never_alert": classification_metrics(test_labels, [0] * len(test)),
                "average_precision": float(average_precision_score(test_labels, test_scores))
                if any(test_labels) else None,
            }
            test_predictions = [
                {key: row[key] for key in ("date", "run_id", "task_id", "failed")}
                | {"failure_score": score, "selected_alert": alert, "baseline_alert": baseline}
                for row, score, alert, baseline in zip(test, test_scores, test_alerts, test_baseline)
            ]
    output_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, output_dir / "model.joblib")
    write_history([
        {"run_id": row["run_id"], "date": row["date"], "task_id": row["task_id"],
         "failed": label, "failure_score": score, "alert_0_5": alert,
         "alert_same_budget": ranked, "baseline_alert": baseline}
        | ({"selected_alert": int(score >= threshold)} if max_alert_rate is not None else {})
        for row, label, score, alert, ranked, baseline in
        zip(validation, labels, scores, rf_alerts, ranked_alerts, fixed_alerts)
    ], output_dir / "validation_predictions.csv")
    if test_predictions:
        write_history(test_predictions, output_dir / "test_predictions.csv")
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/prediction/history.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/prediction/random_forest"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        report = train(args.input, args.output_dir, args.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for key in ("baseline", "random_forest_threshold_0_5", "random_forest_same_alert_count"):
        print(key, json.dumps(report[key]))
    print(f"Average precision: {report['average_precision']}")
    print(f"Rapport et modele : {args.output_dir}")
    print("Test reserve, non evalue.")


if __name__ == "__main__":
    main()
