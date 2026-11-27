"""Explique trois alertes du test avec TreeSHAP, intégré à XGBoost."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
from xgboost import DMatrix

from src.prediction.baseline import load_history
from src.prediction.generate_history import write_history
from src.prediction.train_random_forest import feature_frame


def explain(input_path: Path, model_dir: Path, output: Path) -> list[dict]:
    report = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("model_kind") != "xgboost" or not report.get("test_evaluated"):
        raise ValueError("Un rapport XGBoost avec évaluation de test est requis")
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != report["input_sha256"]:
        raise ValueError("Le CSV d'historique ne correspond pas au rapport")

    with (model_dir / "test_predictions.csv").open(encoding="utf-8", newline="") as stream:
        alerts = [row for row in csv.DictReader(stream) if row["selected_alert"] == "1"]
    alerts.sort(key=lambda row: (-float(row["failure_score"]), row["run_id"]))
    detected = next((row for row in alerts if row["failed"] == "1"), None)
    false_alert = next((row for row in alerts if row["failed"] == "0"), None)
    if detected is None or false_alert is None or len(alerts) < 3:
        raise ValueError("Il faut trois alertes, dont une panne détectée et une fausse alerte")
    third = next(row for row in alerts if row["run_id"] not in
                 {detected["run_id"], false_alert["run_id"]})
    selected = [detected, false_alert, third]

    history = {row["run_id"]: row for row in load_history(input_path)}
    rows = [history[row["run_id"]] for row in selected]
    if any(str(row["failed"]) != alert["failed"] or row["date"] != alert["date"]
           for row, alert in zip(rows, selected)):
        raise ValueError("Les résultats du test ne correspondent pas à l'historique")

    # Charger uniquement un modèle joblib provenant d'une expérience de confiance.
    model = joblib.load(model_dir / "model.joblib")
    preprocessing = model.named_steps["preprocessing"]
    transformed = preprocessing.transform(feature_frame(rows))
    names = (list(transformed.columns) if hasattr(transformed, "columns") else
             list(preprocessing.get_feature_names_out()))
    booster = model.named_steps["classifier"].get_booster()
    contributions = booster.predict(
        DMatrix(transformed, feature_names=booster.feature_names), pred_contribs=True,
    )
    scores = [float(row["failure_score"]) for row in selected]
    if not np.allclose(expit(contributions.sum(axis=1)), scores, rtol=1e-5, atol=1e-7):
        raise ValueError("Les contributions SHAP ne reconstituent pas les scores du test")

    explanations = []
    for alert, values in zip(selected, contributions):
        positive = sorted(
            ((name, float(value)) for name, value in zip(names, values[:-1]) if value > 0),
            key=lambda item: -item[1],
        )
        explanations.append({
            **alert,
            "resultat": "panne_detectee" if alert["failed"] == "1" else "fausse_alerte",
            "shap_base_log_odds": float(values[-1]),
            **{
                f"facteur_{index}": (
                    f"{positive[index - 1][0].split('__', 1)[-1]} ({positive[index - 1][1]:+.4f})"
                    if index <= len(positive) else ""
                )
                for index in range(1, 4)
            },
        })
    write_history(explanations, output)
    return explanations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        explain(args.input, args.model_dir, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Explications SHAP enregistrées dans {args.output}")


if __name__ == "__main__":
    main()
