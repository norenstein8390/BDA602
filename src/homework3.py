import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession

from transformer import Rolling100DayTransform

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


def main():
    transformer = Rolling100DayTransform()
    r100_df = transformer._transform()
    # show table of the rolling 100 result
    r100_df.show()
    return


if __name__ == "__main__":
    sys.exit(main())


'''
brds_query = """SELECT
                      bgs1.batter_id,
                      bgs1.game_id,
                      bgs1.game_date,
                      SUM(bgs2.hit) / SUM(bgs2.AB) AS avg_over_last_100_days
                FROM batter_game_stats bgs1
                JOIN batter_game_stats bgs2
                ON bgs2.game_date
                BETWEEN DATE_SUB(bgs1.game_date, 100) AND DATE_SUB(bgs1.game_date, 1)
                AND bgs1.batter_id = bgs2.batter_id
                WHERE bgs2.AB > 0
                GROUP BY
                      bgs1.batter_id,
                      bgs1.game_id,
                      bgs1.game_date"""

rolling = spark.sql(brds_query)
rolling.show()


brds_df = spark.read.format("jdbc") \
                 .option("url", jdbc_url) \
                 .option("query", brds_query) \
                 .option("user", user) \
                 .option("password", password) \
                 .option("driver", jdbc_driver) \
                 .load()

brds_df.show()
'''
