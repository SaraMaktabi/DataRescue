"""Recherche compacte sur trois fenêtres temporelles de l'entraînement seul."""

import argparse
import json
import math
from pathlib import Path

from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from src.prediction.baseline import chronological_split, load_history
from src.prediction.train_random_forest import feature_frame, train
from src.prediction.train_xgboost import build_xgboost


def temporal_folds(rows: list[dict]) -> list[tuple[list[int], list[int]]]:
    dates = sorted({r["date"] for r in rows})
    if len(dates) < 8:
        raise ValueError("Au moins huit dates d'entraînement sont nécessaires")
    folds = []
    for train_days, valid_days in TimeSeriesSplit(n_splits=3).split(dates):
        past = {dates[i] for i in train_days}
        future = {dates[i] for i in valid_days}
        folds.append(([i for i, r in enumerate(rows) if r["date"] in past],
                      [i for i, r in enumerate(rows) if r["date"] in future]))
    return folds


def class_weight(labels: list[int], mode: str) -> float:
    if set(labels) != {0, 1}:
        raise ValueError("Chaque fenêtre d'entraînement doit contenir les deux classes")
    ratio = labels.count(0) / labels.count(1)
    return {"none": 1.0, "sqrt": math.sqrt(ratio), "ratio": ratio}[mode]


def candidates() -> list[dict]:
    # Référence initiale incluse ; ordre fixe pour départager les égalités.
    return [dict(n_estimators=trees, max_depth=depth, learning_rate=0.05,
                 min_child_weight=5, weight_mode=mode)
            for trees, depth in [(200, 3), (100, 2), (200, 2), (200, 5)]
            for mode in ("ratio", "sqrt", "none")]


def select_parameters(rows: list[dict], seed: int = 42,
                      configurations: list[dict] | None = None) -> dict:
    folds = temporal_folds(rows)
    frame = feature_frame(rows)
    results = []
    for candidate in configurations if configurations is not None else candidates():
        parameters = {k: v for k, v in candidate.items() if k != "weight_mode"}
        fold_scores = []
        weights = []
        for past, future in folds:
            labels = [rows[i]["failed"] for i in past]
            valid_labels = [rows[i]["failed"] for i in future]
            if not any(valid_labels):
                raise ValueError("Chaque fenêtre de validation interne doit contenir une panne")
            weight = class_weight(labels, candidate["weight_mode"])
            model = build_xgboost(seed=seed, scale_pos_weight=weight)
            model.named_steps["classifier"].set_params(**parameters)
            model.fit(frame.iloc[past], labels)
            scores = model.predict_proba(frame.iloc[future])[:, 1]
            fold_scores.append(float(average_precision_score(valid_labels, scores)))
            weights.append(weight)
        results.append({"parameters": candidate, "fold_average_precision": fold_scores,
                        "fold_class_weights": weights,
                        "mean_average_precision": sum(fold_scores) / len(fold_scores)})
    if not results:
        raise ValueError("Aucune configuration à comparer")
    best = max(results, key=lambda result: result["mean_average_precision"])
    selected = {k: v for k, v in best["parameters"].items() if k != "weight_mode"}
    selected["scale_pos_weight"] = class_weight(
        [r["failed"] for r in rows], best["parameters"]["weight_mode"])
    return {
        "selection_set": "train_only", "selection_metric": "mean_average_precision",
        "seed": seed, "best": best, "selected_parameters": selected,
        "candidates": results,
        "folds": [{"train_start": min(rows[i]["date"] for i in past),
                   "train_end": max(rows[i]["date"] for i in past),
                   "validation_start": min(rows[i]["date"] for i in future),
                   "validation_end": max(rows[i]["date"] for i in future),
                   "train_rows": len(past), "validation_rows": len(future)}
                  for past, future in folds],
    }


def search(input_path: Path, output_dir: Path, seed: int = 42) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Dossier existant : {output_dir}")
    training = chronological_split(load_history(input_path))["train"]
    selection = select_parameters(training, seed)
    report = train(input_path, output_dir, seed, model_kind="xgboost",
                   xgb_parameters=selection["selected_parameters"])
    selection["input_sha256"] = report["input_sha256"]
    selection["test_evaluated"] = False
    (output_dir / "search_report.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/prediction/history.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/prediction/xgb_search"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        report = search(args.input, args.output_dir, args.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print("Parametres retenus :", json.dumps(report["model_parameters"]))
    print("Average precision validation :", report["average_precision"])
    print("Meme budget :", json.dumps(report["xgboost_same_alert_count"]))
    print("Seuil 0.5 :", json.dumps(report["xgboost_threshold_0_5"]))
    print("Test reserve, non evalue.")


if __name__ == "__main__":
    main()
