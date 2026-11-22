"""Entraîne Random Forest sur le passé et compare sur la validation uniquement."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.prediction.baseline import (
    chronological_split, classification_metrics, load_history, predict,
)
from src.prediction.generate_history import FEATURE_COLUMNS, write_history


def feature_frame(rows: list[dict]) -> pd.DataFrame:
    """Exclut explicitement cible, résultats et identifiants d'exécution."""
    frame = pd.DataFrame(rows).loc[:, list(FEATURE_COLUMNS)].copy()
    for column in FEATURE_COLUMNS:
        if column != "task_id":
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    return frame


def build_model(seed: int = 42, trees: int = 200) -> Pipeline:
    numeric = [column for column in FEATURE_COLUMNS if column != "task_id"]
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True,
                                  keep_empty_features=True), numeric),
        ("task", OneHotEncoder(handle_unknown="ignore"), ["task_id"]),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
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


def train(input_path: Path, output_dir: Path, seed: int = 42,
          trees: int = 200) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Dossier existant : {output_dir}")
    splits = chronological_split(load_history(input_path))
    training, validation = splits["train"], splits["validation"]
    y_train = [r["failed"] for r in training]
    if set(y_train) != {0, 1}:
        raise ValueError("L'entraînement nécessite des succès et des pannes")
    model = build_model(seed, trees)
    model.fit(feature_frame(training), y_train)
    scores = model.predict_proba(feature_frame(validation))[:, 1].tolist()
    labels = [r["failed"] for r in validation]
    fixed_alerts = [predict(row)["alert"] for row in validation]
    rf_alerts = [int(score >= 0.5) for score in scores]
    budget = sum(fixed_alerts)
    ranked_alerts = top_k_alerts(scores, budget)
    report = {
        "evaluation_set": "validation", "test_evaluated": False,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "seed": seed, "sklearn_version": sklearn.__version__,
        "features": list(FEATURE_COLUMNS),
        "model_parameters": {"n_estimators": trees, "min_samples_leaf": 5,
                             "class_weight": "balanced"},
        "splits": {name: {"rows": len(rows), "start": rows[0]["date"],
                          "end": rows[-1]["date"]} for name, rows in splits.items()},
        "baseline": classification_metrics(labels, fixed_alerts),
        "random_forest_threshold_0_5": classification_metrics(labels, rf_alerts),
        "random_forest_same_alert_count": classification_metrics(labels, ranked_alerts),
        "comparison_alert_budget": budget,
        "average_precision": float(average_precision_score(labels, scores)) if any(labels) else None,
        "failure_prevalence": sum(labels) / len(labels),
        "notes": [
            "Average precision : résumé non interpolé de la courbe précision-rappel, pas aire trapézoïdale.",
            "Comparaison top-k rétrospective au nombre d'alertes de la baseline ; pas une politique en ligne.",
            "Probabilités Random Forest non calibrées ; seuil 0.5 fixé sans optimisation.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, output_dir / "model.joblib")
    write_history([
        {"run_id": row["run_id"], "date": row["date"], "task_id": row["task_id"],
         "failed": label, "failure_score": score, "alert_0_5": alert,
         "alert_same_budget": ranked, "baseline_alert": baseline}
        for row, label, score, alert, ranked, baseline in
        zip(validation, labels, scores, rf_alerts, ranked_alerts, fixed_alerts)
    ], output_dir / "validation_predictions.csv")
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
