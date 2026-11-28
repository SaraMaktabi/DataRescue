"""Interface locale de consultation des expériences de prédiction."""

import csv
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "prediction"
EXPERIMENT = DATA / "xgb_detection"
app = FastAPI(title="DataRescue")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@app.get("/api/dashboard")
def dashboard() -> dict:
    try:
        report = json.loads((EXPERIMENT / "report.json").read_text(encoding="utf-8"))
        predictions = read_csv(EXPERIMENT / "test_predictions.csv")
    except FileNotFoundError as exc:
        raise HTTPException(503, "Le rapport XGBoost et les prédictions du test sont requis.") from exc

    try:
        explanations = read_csv(EXPERIMENT / "shap_alerts.csv")
    except FileNotFoundError:
        explanations = []
    try:
        comparison = read_csv(DATA / "model_comparison" / "comparison.csv")
    except FileNotFoundError:
        comparison = []

    return {
        "period": report["splits"]["test"],
        "metrics": report["test"]["selected_threshold"],
        "threshold": report["operating_point"]["threshold"],
        "predictions": [
            {
                "run_id": row["run_id"], "date": row["date"], "task": row["task_id"],
                "score": float(row["failure_score"]), "alert": row["selected_alert"] == "1",
                "failed": row["failed"] == "1",
            }
            for row in predictions
        ],
        "explanations": {
            row["run_id"]: [row[key] for key in ("facteur_1", "facteur_2", "facteur_3") if row[key]]
            for row in explanations
        },
        "comparison": comparison,
    }


app.mount("/", StaticFiles(directory=ROOT / "frontend" / "dist", html=True, check_dir=False))
