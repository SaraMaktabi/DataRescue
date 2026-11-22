"""Historique synthétique DataRescue, sans dépendance à Airflow ou Spark."""

import argparse
import calendar
import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path


# Profils hypothétiques à ajuster avec les métriques du vrai pipeline.
# task_id, volume habituel, durée habituelle (s), capacité mémoire (Mo)
TASKS = (
    ("bronze_ingest", 100_000, 45, 1024),
    ("accounts_ingest", 20_000, 20, 512),
    ("fx_ingest", 150, 5, 256),
    ("silver_clean", 100_000, 65, 1024),
    ("pseudonymize", 100_000, 35, 1024),
    ("join_accounts", 100_000, 90, 2048),
    ("reconcile", 100_000, 30, 1024),
    ("compute_balances", 20_000, 40, 1024),
    ("aggregate_risk", 20_000, 50, 1024),
    ("export_reporting", 5_000, 15, 512),
)

# Liste explicite pour que les futurs entraînements excluent les résultats.
FEATURE_COLUMNS = (
    "task_id", "input_rows", "file_size_mb", "input_null_rate_pct",
    "arrival_delay_min", "memory_limit_mb", "is_month_end", "is_payday",
    "day_of_week", "previous_duration_s", "previous_peak_memory_mb",
    "previous_retry_count", "previous_failed", "history_count",
    "volume_mean_7d", "volume_delta_pct", "duration_trend_7d_s",
)
OUTCOME_COLUMNS = (
    "failed", "observed_duration_s", "observed_peak_memory_mb",
    "observed_retry_count", "simulated_cause",
)


def generate_history(days: int = 365, seed: int = 42,
                     start_date: date = date(2025, 1, 1)) -> list[dict]:
    """Une ligne = prédiction avant une tâche, puis résultat de cette tâche.

    Les contrôles d'entrée sont supposés disponibles au moment du scoring.
    Les tâches sont simulées indépendamment (pas de propagation dans le DAG).
    """
    if days < 1:
        raise ValueError("days doit être supérieur ou égal à 1")
    rng = random.Random(seed)
    histories: dict[str, list[dict]] = {task[0]: [] for task in TASKS}
    pressure = {task[0]: 1.0 for task in TASKS}
    records = []
    for offset in range(days):
        current = start_date + timedelta(days=offset)
        month_end = int(current.day == calendar.monthrange(current.year, current.month)[1])
        payday = int(current.day == 25)  # Hypothèse du simulateur, configurable ensuite.
        for task_id, base_rows, base_duration, memory_limit in TASKS:
            past = histories[task_id]
            recent = past[-7:]
            previous = past[-1] if past else None
            pressure[task_id] = max(0.6, min(2.2,
                0.85 * pressure[task_id] + 0.15 + rng.gauss(0, 0.10)))
            load = pressure[task_id] * (1 + 0.65 * month_end + 0.35 * payday)
            rows = max(1, round(base_rows * load * rng.lognormvariate(0, 0.18)))
            null_rate = min(30.0, rng.expovariate(1.5)
                            + (rng.uniform(4, 12) if rng.random() < 0.07 else 0))
            delay = rng.expovariate(0.5)
            if rng.random() < 0.08:
                delay += rng.uniform(15, 90)
            mean_volume = sum(r["input_rows"] for r in recent) / len(recent) if recent else None
            volume_delta = 100 * (rows / mean_volume - 1) if mean_volume else None
            trend = ((recent[-1]["observed_duration_s"] - recent[0]["observed_duration_s"])
                     / (len(recent) - 1)) if len(recent) > 1 else None

            # Une demande latente de ressources provoque parfois une panne.
            # Elle n'est pas exposée comme variable disponible avant exécution.
            memory_demand = memory_limit * 0.55 * (rows / base_rows) * rng.lognormvariate(0, 0.18)
            risks = {
                "memory_pressure": 3.0 * max(0, memory_demand / memory_limit - 0.7),
                "source_quality": 0.30 * null_rate,
                "source_delay": 0.035 * delay,
            }
            logit = -3.5 + sum(risks.values()) + rng.gauss(0, 0.45)
            probability = 1 / (1 + math.exp(-max(-30, min(30, logit))))
            failed = int(rng.random() < probability)
            cause = (max(risks, key=risks.get) if max(risks.values()) > 0.6
                     else "unexpected_incident") if failed else "none"
            duration = round(base_duration * (rows / base_rows) ** 0.7
                             * rng.lognormvariate(0, 0.15) * (1.3 if failed else 1), 2)
            row = {
                "date": current.isoformat(),
                "run_id": f"{current.isoformat()}_{task_id}",
                "task_id": task_id,
                "input_rows": rows,
                "file_size_mb": round(rows * rng.uniform(180, 240) / 1_000_000, 4),
                "input_null_rate_pct": round(null_rate, 3),
                "arrival_delay_min": round(delay, 2),
                "memory_limit_mb": memory_limit,
                "is_month_end": month_end,
                "is_payday": payday,
                "day_of_week": current.weekday(),
                "previous_duration_s": previous["observed_duration_s"] if previous else None,
                "previous_peak_memory_mb": previous["observed_peak_memory_mb"] if previous else None,
                "previous_retry_count": previous["observed_retry_count"] if previous else None,
                "previous_failed": previous["failed"] if previous else None,
                "history_count": len(past),
                "volume_mean_7d": round(mean_volume, 2) if mean_volume is not None else None,
                "volume_delta_pct": round(volume_delta, 2) if volume_delta is not None else None,
                "duration_trend_7d_s": round(trend, 2) if trend is not None else None,
                "failed": failed,
                "observed_duration_s": duration,
                "observed_peak_memory_mb": round(min(memory_demand, memory_limit), 2),
                "observed_retry_count": rng.randint(1, 3) if failed else 0,
                "simulated_cause": cause,
            }
            past.append(row)
            records.append(row)
    return records


def write_history(records: list[dict], output: Path) -> None:
    """Écrit sans écraser un historique existant."""
    if not records:
        raise ValueError("L'historique est vide")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--output", type=Path, default=Path("data/prediction/history.csv"))
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days doit être supérieur ou égal à 1")
    records = generate_history(args.days, args.seed, args.start_date)
    try:
        write_history(records, args.output)
    except FileExistsError:
        parser.error("Le fichier existe déjà : choisissez un autre --output.")
    failures = sum(row["failed"] for row in records)
    print(f"{len(records)} lignes, {failures} pannes ({failures / len(records):.1%})")
    print(f"Historique synthétique : {args.output}")


if __name__ == "__main__":
    main()
