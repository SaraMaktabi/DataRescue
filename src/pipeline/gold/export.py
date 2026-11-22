"""
Export des tables Gold en CSV pour le reporting.

Lit les tables Delta Gold et les exporte dans data/reports/<table>.csv

Usage :
    uv run python -m src.pipeline.gold.export
"""
from pathlib import Path

from src.pipeline.paths import data_delta, data_reports
from src.pipeline.spark_session import get_spark
from src.utils.logging import get_logger

log = get_logger("gold_export")

GOLD_TABLES = [
    "daily_account_summary",
    "daily_merchant_summary",
    "reconciliation_check",
]


def export_table(spark, table_name: str) -> dict:
    """Exporte une table Gold en CSV (un seul fichier, header inclus)."""
    src = data_delta("gold", table_name)
    dst = data_reports(f"{table_name}.csv")

    df = spark.read.format("delta").load(str(src))
    count = df.count()

    # coalesce(1) pour produire un seul fichier CSV
    (
        df.coalesce(1)
        .write
        .option("header", "true")
        .mode("overwrite")
        .csv(str(dst.parent / f"{table_name}.tmp"))
    )

    # Récupère le fichier part-* et le renomme en <table>.csv
    import shutil
    tmp_dir = dst.parent / f"{table_name}.tmp"
    part_file = next(tmp_dir.glob("part-*.csv"))
    shutil.move(str(part_file), str(dst))
    shutil.rmtree(tmp_dir)

    log.info("Exporté : %s (%d lignes)", dst, count)
    return {"table": table_name, "rows": count, "path": str(dst)}


def main() -> None:
    log.info("=== Export reporting Gold ===")
    spark = get_spark("Gold-Export")

    data_reports("").mkdir(parents=True, exist_ok=True)

    results = [export_table(spark, t) for t in GOLD_TABLES]
    for r in results:
        log.info("OK : %s", r)

    spark.stop()
    log.info("=== Terminé ===")


if __name__ == "__main__":
    main()