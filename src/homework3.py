import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession

from transformer import Rolling100DayTransform


def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    database = "baseball"
    user = "root"
    password = ""
    server = "localhost"
    port = 3306

    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    bgs_query = """SELECT
                        bc.batter AS batter_id,
                        bc.game_id,
                        DATE(game.local_date) AS game_date,
                        bc.atBat AS AB,
                        bc.Hit AS hit
                  FROM batter_counts bc
                  JOIN game ON bc.game_id = game.game_id"""

    bgs_df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", bgs_query)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    bgs_df.createOrReplaceTempView("batter_game_stats")
    bgs_df.persist(StorageLevel.DISK_ONLY)

    transformer = Rolling100DayTransform()
    r100_df = transformer._transform()
    r100_df.show()
    return


if __name__ == "__main__":
    sys.exit(main())
