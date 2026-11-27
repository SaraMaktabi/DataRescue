"""Compare les trois modèles sur le même historique et le même budget de validation."""

import argparse
from pathlib import Path

from src.prediction.generate_history import write_history
from src.prediction.search_xgboost import search
from src.prediction.train_random_forest import train


def compare(input_path: Path, output_dir: Path, seed: int = 42,
            max_alert_rate: float = 0.15) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=False)
    options = {"max_alert_rate": max_alert_rate, "evaluate_test": True}
    reports = [
        train(input_path, output_dir / "random_forest", seed, **options),
        search(input_path, output_dir / "xgboost", seed, **options),
        train(input_path, output_dir / "isolation_forest", seed,
              model_kind="isolation_forest", **options),
    ]
    results = []
    for report in reports:
        test = report["test"]
        split = report["splits"]["test"]
        operating_point = report["operating_point"]
        results.append({
            "model": report["model_kind"],
            "test_start": split["start"], "test_end": split["end"],
            "validation_alert_budget": max_alert_rate,
            "validation_alert_count": operating_point["alert_count"],
            "threshold": operating_point["threshold"],
            **test["selected_threshold"],
            "average_precision": test["average_precision"],
        })
    write_history(results, output_dir / "comparison.csv")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-alert-rate", type=float, default=0.15)
    args = parser.parse_args()
    try:
        compare(args.input, args.output_dir, args.seed, args.max_alert_rate)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Comparaison enregistrée dans {args.output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
