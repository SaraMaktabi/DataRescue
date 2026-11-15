"""
Ingestion Bronze : lit data/raw/transactions.csv et l'écrit en Delta
sans aucune transformation. Zone brute du pipeline médaillon.

Usage :
    uv run python -m src.pipeline.bronze.ingest
"""
from pathlib import Path

from src.config_loader import CONFIG
from src.pipeline.spark_session import get_spark
from src.utils.logging import get_logger
from src.pipeline.paths import data_raw, data_delta

log = get_logger("bronze_ingest")

RAW_PATH = data_raw("transactions.csv")
BRONZE_PATH = data_delta("bronze", "transactions")


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Fichier source introuvable : {RAW_PATH}")

    log.info("=== Ingestion Bronze ===")
    log.info("Source : %s", RAW_PATH)
    log.info("Destination : %s", BRONZE_PATH)

    spark = get_spark("Bronze-Ingest")

    # Lecture brute : on laisse Spark inférer le schéma pour l'instant.
    # En production on définirait un schéma explicite [citation:2][citation:14].
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(RAW_PATH))
    )

    log.info("Lignes lues : %d", df.count())
    log.info("Colonnes : %s", df.columns)

    # Écriture Delta en mode overwrite (idempotent : on peut relancer)
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .save(str(BRONZE_PATH))
    )

    log.info("Écrit : %s", BRONZE_PATH)

    # Vérification rapide
    count = spark.read.format("delta").load(str(BRONZE_PATH)).count()
    log.info("Vérification : %d lignes dans la table Delta", count)

    spark.stop()
    log.info("=== Terminé ===")


if __name__ == "__main__":
    main()