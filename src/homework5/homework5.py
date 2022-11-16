import sys

from homework4_main import Homework4ReportMaker
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

    df["home_avg"] = df["home_avg"].astype("float")
    df["home_starter_baa"] = df["home_starter_baa"].astype("float")
    df["away_avg"] = df["away_avg"].astype("float")
    df["away_starter_baa"] = df["away_starter_baa"].astype("float")

    predictors = [
        "home_avg",
        "home_obp",
        "home_slg",
        "home_starter_whip",
        "home_starter_baa",
        "home_starter_k",
        "away_avg",
        "away_obp",
        "away_slg",
        "away_starter_whip",
        "away_starter_baa",
        "away_starter_k",
    ]
    response = "HomeTeamWins"

    hw4_report_maker = Homework4ReportMaker(df, predictors, response)
    hw4_report_maker.make_plots_rankings()


if __name__ == "__main__":
    sys.exit(main())
