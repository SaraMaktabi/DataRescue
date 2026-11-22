"""Découpage chronologique et référence de seuils fixes, sans apprentissage."""

import argparse
import csv
import json
import math
from datetime import date
from pathlib import Path

from src.prediction.generate_history import FEATURE_COLUMNS, write_history


# Hypothèses initiales, fixées sans consulter les résultats du test.
THRESHOLDS = {
    "input_null_rate_pct": 5.0,
    "arrival_delay_min": 30.0,
    "absolute_volume_delta_pct": 50.0,
    "previous_memory_ratio": 0.90,
}
NUMERIC_COLUMNS = set(FEATURE_COLUMNS) - {"task_id"}
OPTIONAL_COLUMNS = {
    "previous_duration_s", "previous_peak_memory_mb", "previous_retry_count",
    "previous_failed", "volume_mean_7d", "volume_delta_pct", "duration_trend_7d_s",
}


def load_history(path: Path) -> list[dict]:
    """Charge uniquement les entrées autorisées, identifiants et cible."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"date", "run_id", "failed", *FEATURE_COLUMNS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Colonnes absentes : {', '.join(sorted(missing))}")
        records = []
        seen = set()
        for line, source in enumerate(reader, start=2):
            try:
                row = {key: source[key] for key in ("date", "run_id", *FEATURE_COLUMNS, "failed")}
                row["date"] = date.fromisoformat(row["date"]).isoformat()
                if not row["run_id"] or not row["task_id"]:
                    raise ValueError("identifiant vide")
                if row["run_id"] in seen:
                    raise ValueError("run_id dupliqué")
                seen.add(row["run_id"])
                if row["failed"] not in {"0", "1"}:
                    raise ValueError("failed doit valoir 0 ou 1")
                row["failed"] = int(row["failed"])
                for key in NUMERIC_COLUMNS:
                    if row[key] == "" and key in OPTIONAL_COLUMNS:
                        row[key] = None
                    else:
                        row[key] = float(row[key])
                        if not math.isfinite(row[key]):
                            raise ValueError(f"{key} doit être fini")
                if row["memory_limit_mb"] <= 0:
                    raise ValueError("memory_limit_mb doit être positif")
                if not 0 <= row["input_null_rate_pct"] <= 100:
                    raise ValueError("taux de nulls hors de [0, 100]")
                records.append(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Ligne {line} : {exc}") from exc
    if not records:
        raise ValueError("Historique vide")
    return records


def chronological_split(records: list[dict], train_ratio: float = 0.70,
                        validation_ratio: float = 0.15) -> dict[str, list[dict]]:
    """Garde toutes les tâches d'une même date dans le même ensemble."""
    if not (0 < train_ratio < 1 and 0 < validation_ratio < 1
            and train_ratio + validation_ratio < 1):
        raise ValueError("Ratios positifs requis, avec une somme inférieure à 1")
    dates = sorted({row["date"] for row in records})
    train_days = int(len(dates) * train_ratio)
    validation_days = int(len(dates) * validation_ratio)
    if min(train_days, validation_days, len(dates) - train_days - validation_days) < 1:
        raise ValueError("Pas assez de dates pour trois ensembles non vides")
    train_dates = set(dates[:train_days])
    validation_dates = set(dates[train_days:train_days + validation_days])
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for row in sorted(records, key=lambda r: (r["date"], r["task_id"], r["run_id"])):
        name = ("train" if row["date"] in train_dates else
                "validation" if row["date"] in validation_dates else "test")
        splits[name].append(row)
    return splits


def predict(row: dict) -> dict:
    """Une alerte si au moins un seuil est atteint ; ce n'est pas une probabilité."""
    reasons = []
    if row["input_null_rate_pct"] >= THRESHOLDS["input_null_rate_pct"]:
        reasons.append("high_null_rate")
    if row["arrival_delay_min"] >= THRESHOLDS["arrival_delay_min"]:
        reasons.append("late_source")
    delta = row["volume_delta_pct"]
    if delta is not None and abs(delta) >= THRESHOLDS["absolute_volume_delta_pct"]:
        reasons.append("unusual_volume")
    memory = row["previous_peak_memory_mb"]
    if memory is not None and memory / row["memory_limit_mb"] >= THRESHOLDS["previous_memory_ratio"]:
        reasons.append("previous_memory_pressure")
    return {"alert": int(bool(reasons)), "reasons": reasons}


def classification_metrics(labels: list[int], alerts: list[int]) -> dict:
    if not labels or len(labels) != len(alerts):
        raise ValueError("Listes non vides de même longueur requises")
    if not set(labels + alerts) <= {0, 1}:
        raise ValueError("Labels et alertes binaires requis")
    tp = sum(y == 1 and a == 1 for y, a in zip(labels, alerts))
    fp = sum(y == 0 and a == 1 for y, a in zip(labels, alerts))
    fn = sum(y == 1 and a == 0 for y, a in zip(labels, alerts))
    tn = sum(y == 0 and a == 0 for y, a in zip(labels, alerts))
    return {
        "rows": len(labels), "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "alert_rate": (tp + fp) / len(labels),
    }


def run(input_path: Path, output_dir: Path) -> dict:
    records = load_history(input_path)
    splits = chronological_split(records)
    validation = splits["validation"]
    predictions = []
    for row in validation:
        result = predict(row)
        predictions.append({
            "date": row["date"], "run_id": row["run_id"], "task_id": row["task_id"],
            "failed": row["failed"], "alert": result["alert"],
            "reasons": "|".join(result["reasons"]),
        })
    labels = [r["failed"] for r in validation]
    report = {
        "evaluation_set": "validation",
        "thresholds": THRESHOLDS,
        "splits": {name: {"rows": len(rows), "days": len({r["date"] for r in rows}),
                           "start": rows[0]["date"], "end": rows[-1]["date"]}
                   for name, rows in splits.items()},
        "baseline": classification_metrics(labels, [r["alert"] for r in predictions]),
        "never_alert": classification_metrics(labels, [0] * len(labels)),
        "test_evaluated": False,
    }
    # Refus d'écraser une expérience existante, avant toute écriture.
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in splits.items():
        write_history(rows, output_dir / f"{name}.csv")
    write_history(predictions, output_dir / "validation_predictions.csv")
    with (output_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/prediction/history.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/prediction/baseline"))
    args = parser.parse_args()
    try:
        report = run(args.input, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for name, info in report["splits"].items():
        print(f"{name}: {info['rows']} lignes, {info['start']} -> {info['end']}")
    print("Baseline sur validation :")
    print(json.dumps(report["baseline"], indent=2))
    print(f"Rapport : {args.output_dir / 'report.json'}")
    print("Test reserve : aucune evaluation sur le test.")


if __name__ == "__main__":
    main()
