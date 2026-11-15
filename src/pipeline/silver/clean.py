"""
Silver : nettoie et valide la table Bronze, écrit la table Silver.

Transformations :
- typage explicite des colonnes
- suppression des lignes avec transaction_id ou amount null
- ajout de colonnes de traçabilité (_ingested_at, _source)
- filtrage des montants aberrants (|amount| > 10 000)

Usage :
    uv run python -m src.pipeline.silver.clean
"""
from pathlib import Path

from pyspark.sql import functions as F

from src.pipeline.spark_session import get_spark
from src.utils.logging import get_logger

log = get_logger("silver_clean")

from src.pipeline.paths import data_delta

BRONZE_PATH = data_delta("bronze", "transactions")
SILVER_PATH = data_delta("silver", "transactions")


def main() -> None:
    log.info("=== Nettoyage Silver ===")

    spark = get_spark("Silver-Clean")
    df = spark.read.format("delta").load(str(BRONZE_PATH))

    log.info("Lignes Bronze : %d", df.count())

    # 1. Typage explicite
    df = (
        df
        .withColumn("transaction_id", F.col("transaction_id").cast("long"))
        .withColumn("account_id", F.col("account_id").cast("long"))
        .withColumn("card_id", F.col("card_id").cast("long"))
        .withColumn("amount", F.col("amount").cast("double"))
        .withColumn("timestamp", F.col("timestamp").cast("timestamp"))
        .withColumn("merchant_id", F.col("merchant_id").cast("long"))
        .withColumn("mcc", F.col("mcc").cast("int"))
    )

    # 2. Filtrer les lignes invalides
    before = df.count()
    df = df.filter(
        F.col("transaction_id").isNotNull()
        & F.col("amount").isNotNull()
        & F.col("timestamp").isNotNull()
    )
    after = df.count()
    log.info("Lignes supprimées (nulls) : %d", before - after)

    # 3. Filtrer les montants aberrants (garde-fou simple)
    before = df.count()
    df = df.filter(F.abs(F.col("amount")) <= 10_000)
    after = df.count()
    log.info("Lignes supprimées (montant > 10k) : %d", before - after)

    # 4. Ajouter colonnes de traçabilité
    df = (
        df
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source", F.lit("transactions.csv"))
    )

    # 5. Écriture Silver (overwrite pour idempotence)
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .save(str(SILVER_PATH))
    )

    final_count = spark.read.format("delta").load(str(SILVER_PATH)).count()
    log.info("Écrit : %s", SILVER_PATH)
    log.info("Vérification : %d lignes dans Silver", final_count)

    spark.stop()
    log.info("=== Terminé ===")


if __name__ == "__main__":
    main()