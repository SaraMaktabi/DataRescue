"""
DAG de clôture journalière DataRescue.

10 tâches :
  1. extract_transactions          (capteur source)
  2. bronze_ingest                 (ingestion brute)
  3. silver_clean                  (nettoyage)
  4. quality_check_transactions    (contrôles qualité)
  5. gold_daily_account_summary    (agrégats comptes)
  6. gold_daily_merchant_summary   (agrégats marchands)
  7. gold_reconciliation_check     (réconciliation débits/crédits)
  8. lineage_update                (graphe NetworkX)
  9. report_export                 (export reporting)
 10. notify_completion             (clôture + audit)

Exécution manuelle uniquement (schedule=None) pour le PFA.
"""
from __future__ import annotations

import datetime as dt

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

from src.utils.logging import get_logger

import sys
from pathlib import Path

# Ajoute /opt/airflow au chemin pour que Python trouve le package 'src'
project_root = Path(__file__).parent.parent  # remonte de dags/ vers /opt/airflow
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logging import get_logger

log = get_logger("dag_daily_close")


# ---------------------------------------------------------------------------
# Tâches
# ---------------------------------------------------------------------------

def extract_transactions(**context) -> dict:
    """Vérifie que le fichier source est présent et lisible."""
    from src.pipeline.paths import data_raw
    import pandas as pd

    path = data_raw("transactions.csv")
    if not path.exists():
        raise FileNotFoundError(f"Fichier source introuvable : {path}")

    df = pd.read_csv(path, nrows=1)
    log.info("Source OK : %s | colonnes=%s", path, list(df.columns))
    return {"source": str(path), "columns": list(df.columns)}


def bronze_ingest(**context) -> dict:
    """Ingestion brute CSV -> Delta bronze."""
    from src.pipeline.spark_session import get_spark
    from src.pipeline.paths import data_raw, data_delta

    spark = get_spark("DAG-Bronze")
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(data_raw("transactions.csv")))
    )
    count = df.count()
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .save(str(data_delta("bronze", "transactions")))
    )
    log.info("Bronze : %d lignes", count)
    spark.stop()
    return {"bronze_count": count}


def silver_clean(**context) -> dict:
    """Nettoyage Bronze -> Silver."""
    from pyspark.sql import functions as F
    from src.pipeline.spark_session import get_spark
    from src.pipeline.paths import data_delta

    spark = get_spark("DAG-Silver")
    df = spark.read.format("delta").load(str(data_delta("bronze", "transactions")))

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
    df = df.filter(
        F.col("transaction_id").isNotNull()
        & F.col("amount").isNotNull()
        & F.col("timestamp").isNotNull()
    )
    df = df.filter(F.abs(F.col("amount")) <= 10_000)
    df = (
        df
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source", F.lit("transactions.csv"))
    )
    count = df.count()
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .save(str(data_delta("silver", "transactions")))
    )
    log.info("Silver : %d lignes", count)
    spark.stop()
    return {"silver_count": count}


def quality_check_transactions(**context) -> dict:
    """Contrôles qualité : unicité, nulls, montants."""
    from pyspark.sql import functions as F
    from src.pipeline.spark_session import get_spark
    from src.pipeline.paths import data_delta

    spark = get_spark("DAG-Quality")
    df = spark.read.format("delta").load(str(data_delta("silver", "transactions")))

    total = df.count()

    # Contrôle 1 : unicité de transaction_id
    duplicates = df.groupBy("transaction_id").count().filter(F.col("count") > 1).count()
    if duplicates > 0:
        raise ValueError(f"Doublons détectés : {duplicates} transaction_id dupliqués")

    # Contrôle 2 : nulls sur colonnes critiques
    nulls = df.filter(
        F.col("transaction_id").isNull()
        | F.col("amount").isNull()
        | F.col("timestamp").isNull()
    ).count()
    if nulls > 0:
        raise ValueError(f"Nulls détectés : {nulls} lignes avec nulls critiques")

    log.info("Qualité OK : %d lignes, 0 doublon, 0 null", total)
    spark.stop()
    return {"total": total, "duplicates": duplicates, "nulls": nulls}


def gold_daily_account_summary(**context) -> dict:
    """Agrégats par compte et par jour."""
    from src.pipeline.gold.aggregate import build_daily_account_summary
    from src.pipeline.paths import data_delta
    from src.pipeline.spark_session import get_spark
    from src.utils.logging import get_logger

    log = get_logger("dag_gold_accounts")
    spark = get_spark("DAG-Gold-Accounts")

    silver = spark.read.format("delta").load(str(data_delta("silver", "transactions")))
    summary = build_daily_account_summary(silver)
    count = summary.count()

    (
        summary.write
        .format("delta")
        .mode("overwrite")
        .save(str(data_delta("gold", "daily_account_summary")))
    )

    log.info("Gold daily_account_summary : %d lignes", count)
    spark.stop()
    return {"gold_account_summary_count": count}


def gold_daily_merchant_summary(**context) -> dict:
    """Agrégats par marchand et par jour."""
    from src.pipeline.gold.aggregate import build_daily_merchant_summary
    from src.pipeline.paths import data_delta
    from src.pipeline.spark_session import get_spark
    from src.utils.logging import get_logger

    log = get_logger("dag_gold_merchants")
    spark = get_spark("DAG-Gold-Merchants")

    silver = spark.read.format("delta").load(str(data_delta("silver", "transactions")))
    merchant = build_daily_merchant_summary(silver)
    count = merchant.count()

    (
        merchant.write
        .format("delta")
        .mode("overwrite")
        .save(str(data_delta("gold", "daily_merchant_summary")))
    )

    log.info("Gold daily_merchant_summary : %d lignes", count)
    spark.stop()
    return {"gold_merchant_summary_count": count}


def gold_reconciliation_check(**context) -> dict:
    """Contrôle débits = crédits par jour (règle R7)."""
    from src.pipeline.gold.aggregate import build_reconciliation_check
    from src.pipeline.paths import data_delta
    from src.pipeline.spark_session import get_spark
    from src.utils.logging import get_logger
    from pyspark.sql import functions as F

    log = get_logger("dag_gold_recon")
    spark = get_spark("DAG-Gold-Recon")

    silver = spark.read.format("delta").load(str(data_delta("silver", "transactions")))
    recon = build_reconciliation_check(silver)
    count = recon.count()
    non_reconciled = recon.filter(~F.col("is_reconciled")).count()

    (
        recon.write
        .format("delta")
        .mode("overwrite")
        .save(str(data_delta("gold", "reconciliation_check")))
    )

    log.info("Gold reconciliation_check : %d jours, %d non réconciliés", count, non_reconciled)
    spark.stop()
    return {"days": count, "non_reconciled": non_reconciled}


def lineage_update(**context) -> dict:
    """STUB — à implémenter : mise à jour du graphe NetworkX."""
    log.info("TODO: lineage_update (stub)")
    return {"status": "stub"}


def report_export(**context) -> dict:
    """Export des tables Gold en CSV pour le reporting."""
    from src.pipeline.gold.export import main as export_main
    export_main()
    return {"status": "exported"}


def notify_completion(**context) -> dict:
    """Clôture : log d'audit final."""
    run_date = context.get("ds", "unknown")
    log.info("Pipeline clôturé pour %s", run_date)
    return {"status": "completed", "date": run_date}


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="datarescue_daily_close",
    description="Clôture journalière bancaire : Bronze -> Silver -> Gold -> Reporting",
    start_date=dt.datetime(2024, 1, 1),
    schedule=None,  # exécution manuelle uniquement (PFA)
    catchup=False,
    tags=["datarescue", "banking", "medallion"],
    default_args={
        "owner": "datarescue",
        "retries": 1,
        "retry_delay": dt.timedelta(minutes=2),
    },
) as dag:

    t1_extract = PythonOperator(task_id="extract_transactions", python_callable=extract_transactions)
    t2_bronze = PythonOperator(task_id="bronze_ingest", python_callable=bronze_ingest)
    t3_silver = PythonOperator(task_id="silver_clean", python_callable=silver_clean)
    t4_quality = PythonOperator(task_id="quality_check_transactions", python_callable=quality_check_transactions)

    t5_gold_accounts = PythonOperator(task_id="gold_daily_account_summary", python_callable=gold_daily_account_summary)
    t6_gold_merchants = PythonOperator(task_id="gold_daily_merchant_summary", python_callable=gold_daily_merchant_summary)

    t7_reconciliation = PythonOperator(task_id="gold_reconciliation_check", python_callable=gold_reconciliation_check)

    t8_lineage = PythonOperator(task_id="lineage_update", python_callable=lineage_update)
    t9_report = PythonOperator(task_id="report_export", python_callable=report_export)
    t10_notify = PythonOperator(task_id="notify_completion", python_callable=notify_completion)

    # Dépendances
    t1_extract >> t2_bronze >> t3_silver >> t4_quality
    t4_quality >> [t5_gold_accounts, t6_gold_merchants]
    [t5_gold_accounts, t6_gold_merchants] >> t7_reconciliation
    t7_reconciliation >> [t8_lineage, t9_report]
    [t8_lineage, t9_report] >> t10_notify