"""
Agrégats Gold : tables métier calculées à partir de Silver.

Tables produites :
- gold/daily_account_summary   : agrégats par compte et par jour
- gold/daily_merchant_summary  : agrégats par marchand et par jour (à venir)
- gold/reconciliation_check    : contrôle débits = crédits (à venir)

Usage :
    uv run python -m src.pipeline.gold.aggregate
"""
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.pipeline.paths import data_delta
from src.pipeline.spark_session import get_spark
from src.utils.logging import get_logger

log = get_logger("gold_aggregate")


def build_daily_account_summary(silver: DataFrame) -> DataFrame:
    """
    Agrégats par compte et par jour.

    Colonnes de sortie :
    - account_id
    - date
    - total_debits   (somme des montants < 0, en valeur absolue)
    - total_credits  (somme des montants > 0)
    - net_flow       (credits - debits)
    - transaction_count
    """
    df = silver.withColumn("date", F.to_date("timestamp"))

    summary = (
        df.groupBy("account_id", "date")
        .agg(
            F.sum(F.when(F.col("amount") < 0, F.abs(F.col("amount"))).otherwise(0.0)).alias("total_debits"),
            F.sum(F.when(F.col("amount") > 0, F.col("amount")).otherwise(0.0)).alias("total_credits"),
            F.count("*").alias("transaction_count"),
        )
        .withColumn("net_flow", F.col("total_credits") - F.col("total_debits"))
        .select(
            "account_id",
            "date",
            F.round("total_debits", 2).alias("total_debits"),
            F.round("total_credits", 2).alias("total_credits"),
            F.round("net_flow", 2).alias("net_flow"),
            "transaction_count",
        )
    )
    return summary

def build_reconciliation_check(silver: DataFrame) -> DataFrame:
    """
    Contrôle de réconciliation : débits = crédits par jour.

    Colonnes de sortie :
    - date
    - total_debits
    - total_credits
    - difference     (credits - debits)
    - is_reconciled  (true si difference == 0)
    """
    df = silver.withColumn("date", F.to_date("timestamp"))

    recon = (
        df.groupBy("date")
        .agg(
            F.sum(F.when(F.col("amount") < 0, F.abs(F.col("amount"))).otherwise(0.0)).alias("total_debits"),
            F.sum(F.when(F.col("amount") > 0, F.col("amount")).otherwise(0.0)).alias("total_credits"),
        )
        .withColumn("difference", F.round(F.col("total_credits") - F.col("total_debits"), 2))
        .withColumn("is_reconciled", F.col("difference") == 0)
        .select(
            "date",
            F.round("total_debits", 2).alias("total_debits"),
            F.round("total_credits", 2).alias("total_credits"),
            "difference",
            "is_reconciled",
        )
        .orderBy("date")
    )
    return recon

def build_daily_merchant_summary(silver: DataFrame) -> DataFrame:
    """
    Agrégats par marchand et par jour.

    Colonnes de sortie :
    - merchant_id
    - merchant_city
    - merchant_state
    - date
    - total_amount
    - transaction_count
    - avg_amount
    """
    df = silver.withColumn("date", F.to_date("timestamp"))

    merchant = (
        df.groupBy("merchant_id", "merchant_city", "merchant_state", "date")
        .agg(
            F.sum(F.abs(F.col("amount"))).alias("total_amount"),
            F.count("*").alias("transaction_count"),
            F.avg(F.abs(F.col("amount"))).alias("avg_amount"),
        )
        .select(
            "merchant_id",
            "merchant_city",
            "merchant_state",
            "date",
            F.round("total_amount", 2).alias("total_amount"),
            "transaction_count",
            F.round("avg_amount", 2).alias("avg_amount"),
        )
        .orderBy(F.desc("date"), F.desc("total_amount"))
    )
    return merchant

def main() -> None:
    log.info("=== Agrégats Gold : daily_account_summary ===")

    spark = get_spark("Gold-Aggregate")
    silver = spark.read.format("delta").load(str(data_delta("silver", "transactions")))

    silver_count = silver.count()
    log.info("Lignes Silver lues : %d", silver_count)

    summary = build_daily_account_summary(silver)
    out_count = summary.count()
    log.info("Lignes Gold produites : %d", out_count)

    out_path = data_delta("gold", "daily_account_summary")

    # --- Réconciliation ---
    log.info("=== Agrégats Gold : reconciliation_check ===")
    recon = build_reconciliation_check(silver)
    recon_count = recon.count()
    log.info("Lignes réconciliation : %d", recon_count)

    recon_path = data_delta("gold", "reconciliation_check")
    (
        recon.write
        .format("delta")
        .mode("overwrite")
        .save(str(recon_path))
    )
    log.info("Écrit : %s", recon_path)

    # Aperçu + alerte si non réconcilié
    recon.show(31, truncate=False)
    non_reconciled = recon.filter(~F.col("is_reconciled")).count()
    if non_reconciled > 0:
        log.warning("ATTENTION : %d jours non réconciliés", non_reconciled)
    else:
        log.info("Tous les jours sont réconciliés")

        # --- Marchands ---
    log.info("=== Agrégats Gold : daily_merchant_summary ===")
    merchant = build_daily_merchant_summary(silver)
    merchant_count = merchant.count()
    log.info("Lignes marchands : %d", merchant_count)

    merchant_path = data_delta("gold", "daily_merchant_summary")
    (
        merchant.write
        .format("delta")
        .mode("overwrite")
        .save(str(merchant_path))
    )
    log.info("Écrit : %s", merchant_path)
    merchant.show(5, truncate=False)

    
    (
        summary.write
        .format("delta")
        .mode("overwrite")
        .save(str(out_path))
    )
    log.info("Écrit : %s", out_path)

    # Aperçu
    summary.orderBy(F.desc("date"), F.desc("total_credits")).show(5, truncate=False)

    spark.stop()
    log.info("=== Terminé ===")


if __name__ == "__main__":
    main()