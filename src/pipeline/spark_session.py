
"""Crée une SparkSession configurée pour Delta Lake."""

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


def get_spark(app_name: str = "DataRescue") -> SparkSession:
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0")  # ← toujours
        .config("spark.sql.shuffle.partitions", "4")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

    # ANCIEN CODE DE SARA (conservé en commentaire)
    # En local, télécharger Delta.
    # Dans Docker, supposer que les JAR sont déjà présents.
    #
    # if os.environ.get("IN_DOCKER") != "true":
    #     builder = builder.config(
    #         "spark.jars.packages",
    #         "io.delta:delta-spark_2.12:3.1.0"
    #     )

    # CORRECTION : configurer les dépendances JVM Delta Lake
    # via le package delta-spark installé.
    builder = configure_spark_with_delta_pip(builder)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    return spark
