
"""DAG DataRescue Banque — Pipeline de transactions."""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="datarescue_daily_close",
    description="Pipeline bancaire Bronze → Silver → Gold",
    start_date=datetime(2026, 11, 15),
    schedule=None,  # Exécution manuelle pendant le développement
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    tags=["datarescue", "banking"],
) as dag:

    bronze_ingest = BashOperator(
        task_id="bronze_ingest",
        bash_command=(
            "cd /opt/airflow && "
            "mkdir -p /tmp/spark-ivy && "
            "IN_DOCKER=false PYTHONPATH=/opt/airflow "
            "python -m src.pipeline.bronze.ingest"
        ),
    )

    silver_clean = BashOperator(
        task_id="silver_clean",
        bash_command=(
            "cd /opt/airflow && "
            "mkdir -p /tmp/spark-ivy && "
            "IN_DOCKER=false PYTHONPATH=/opt/airflow "
            "python -m src.pipeline.silver.clean"
        ),
    )

    bronze_ingest >> silver_clean
