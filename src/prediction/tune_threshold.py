"""Choisit un seuil sur les prédictions de validation, sans réentraîner le modèle."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from src.prediction.baseline import classification_metrics
from src.prediction.generate_history import write_history


def compare_thresholds(labels: list[int], scores: list[float],
                       max_alert_rate: float = 0.15) -> tuple[list[dict], dict]:
    if not math.isfinite(max_alert_rate) or not 0 <= max_alert_rate <= 1:
        raise ValueError("Le budget doit être compris entre 0 et 1")
    if not labels or len(labels) != len(scores) or not set(labels) <= {0, 1}:
        raise ValueError("Labels binaires et scores non vides de même longueur requis")
    if not any(labels):
        raise ValueError("La validation doit contenir au moins une panne pour choisir le rappel")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Scores finis compris entre 0 et 1 requis")
    budget = math.floor(len(labels) * max_alert_rate)
    # Seuil > 1 : aucune alerte, y compris lorsque certains scores valent 1.
    thresholds = sorted({0.0, 0.5, math.nextafter(1.0, math.inf), *scores})
    results = []
    for threshold in thresholds:
        alerts = [int(score >= threshold) for score in scores]
        metrics = classification_metrics(labels, alerts)
        count = sum(alerts)
        results.append({"threshold": threshold, "alert_count": count,
                        "within_budget": count <= budget, **metrics})
    # Rappel maximal, puis moins de fausses alertes, puis seuil le plus élevé.
    selected = max((r for r in results if r["within_budget"]),
                   key=lambda r: (r["recall"], -r["false_positives"], r["threshold"]))
    return results, selected


def tune(model_dir: Path, output_dir: Path, max_alert_rate: float = 0.15) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Dossier existant : {output_dir}")
    metadata = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
    if metadata.get("evaluation_set") != "validation" or metadata.get("test_evaluated") is not False:
        raise ValueError("Un rapport de validation avec test réservé est requis")
    with (model_dir / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    info = metadata["splits"]["validation"]
    if len(predictions) != info["rows"] or len({r["run_id"] for r in predictions}) != len(predictions):
        raise ValueError("Nombre de prédictions incohérent ou identifiants dupliqués")
    if any(not info["start"] <= r["date"] <= info["end"] for r in predictions):
        raise ValueError("Prédictions hors période de validation")
    if any(r["failed"] not in {"0", "1"} for r in predictions):
        raise ValueError("Labels binaires requis")
    labels = [int(r["failed"]) for r in predictions]
    scores = [float(r["failure_score"]) for r in predictions]
    results, selected = compare_thresholds(labels, scores, max_alert_rate)
    report = {
        "selection_set": "validation", "test_evaluated": False,
        "model_dir": str(model_dir.resolve()), "model_kind": metadata.get("model_kind"),
        "model_sha256": hashlib.sha256((model_dir / "model.joblib").read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256((model_dir / "validation_predictions.csv").read_bytes()).hexdigest(),
        "input_sha256": metadata["input_sha256"],
        "max_alert_rate": max_alert_rate,
        "max_alert_count": math.floor(len(labels) * max_alert_rate),
        "decision_rule": "score >= threshold",
        "selection_rule": "max recall, min false positives, max threshold",
        "selected": selected,
        "threshold_0_5": next(r for r in results if r["threshold"] == 0.5),
        "notes": [
            "Performances mesurées sur les données de sélection, pas une évaluation indépendante.",
            "Budget respecté sur cette validation uniquement ; le taux futur peut changer.",
            "Le réglage du seuil ne change ni les scores ni l'Average Precision du modèle.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_history(results, output_dir / "threshold_comparison.csv")
    write_history([r | {"selected_alert": int(score >= selected["threshold"])}
                   for r, score in zip(predictions, scores)],
                  output_dir / "validation_predictions.csv")
    (output_dir / "threshold.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-alert-rate", type=float, default=0.15)
    args = parser.parse_args()
    try:
        report = tune(args.model_dir, args.output_dir, args.max_alert_rate)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print("Seuil 0.5 :", json.dumps(report["threshold_0_5"]))
    print("Seuil choisi :", json.dumps(report["selected"]))
    print(f"Budget maximum : {report['max_alert_count']} alertes")
    print("Selection sur validation uniquement. Test non evalue.")


if __name__ == "__main__":
    main()
