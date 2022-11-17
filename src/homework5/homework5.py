import sys

from homework4_main import Homework4ReportMaker
from midterm_main import MidtermReportMaker
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from sklearn import svm, tree
from sklearn.model_selection import train_test_split


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

    df = pyspark_df.toPandas().dropna().reset_index()

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

    midterm_report_maker = MidtermReportMaker(df, predictors, response)
    midterm_report_maker.make_correlations_bruteforce()

    X = df[predictors]
    Y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=100
    )

    decision_tree = tree.DecisionTreeClassifier()
    decision_tree = decision_tree.fit(X_train, y_train)
    decision_tree_score = decision_tree.score(X_test, y_test)
    print(f"decision tree score: {decision_tree_score}")

    my_svm = svm.SVC()
    my_svm = my_svm.fit(X_train, y_train)
    svm_score = my_svm.score(X_test, y_test)
    print(f"svm score: {svm_score}")

    # SVM is better


if __name__ == "__main__":
    sys.exit(main())
