"""Crée une SparkSession configurée pour Delta Lake."""
import os

from pyspark.sql import SparkSession


def get_spark(app_name: str = "DataRescue") -> SparkSession:
    """Retourne une SparkSession avec Delta Lake activé."""
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "4")
    )

    # En local (Windows), on a besoin de télécharger Delta.
    # Dans le conteneur, Delta est déjà dans l'image → on ne le retélécharge pas.
    if os.environ.get("IN_DOCKER") != "true":
        builder = builder.config(
            "spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0"
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")   
    return spark