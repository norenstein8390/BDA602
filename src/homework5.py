import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    database = "baseball"
    user = "root"
    password = ""
    server = "localhost"
    port = 3306

    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    sql_query = """SELECT * FROM hw5_feature_stats"""

    pyspark_df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql_query)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    pyspark_df.createOrReplaceTempView("batter_game_stats")
    pyspark_df.persist(StorageLevel.DISK_ONLY)

    df = pyspark_df.toPandas()
    print(type(df))


if __name__ == "__main__":
    sys.exit(main())
