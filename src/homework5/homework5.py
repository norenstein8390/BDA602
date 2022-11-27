import os
import sys
import webbrowser

from homework4_main import Homework4ReportMaker
from midterm_main import MidtermReportMaker
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from sklearn import svm, tree
from sklearn.model_selection import train_test_split


def models_test(df, predictors, response, models):
    X = df[predictors]
    Y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=100
    )
    output = "\n\n<h2>Model Scores</h2>"
    best_name = ""
    best_score = 0

    for i in range(len(models)):
        model = models[i]
        model = model.fit(X_train, y_train)
        model_name = str(model)
        model_score = model.score(X_test, y_test)

        if model_score > best_score:
            best_score = model_score
            best_name = model_name

        output += f"\n<h3>* {model_name} Score: {model_score}</h3>"

    output += (
        f"\n\n<h3>The best model tested was {best_name} (Score = {best_score})</h3>"
    )

    return output


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
    df["home_baa"] = df["home_baa"].astype("float")
    df["away_avg"] = df["away_avg"].astype("float")
    df["away_baa"] = df["away_baa"].astype("float")

    predictors = [
        "home_avg",
        "home_obp",
        "home_slg",
        "home_whip",
        "home_baa",
        "home_pitcher_k",
        "away_avg",
        "away_obp",
        "away_slg",
        "away_whip",
        "away_baa",
        "away_pitcher_k",
    ]
    response = "HomeTeamWins"

    hw4_report_maker = Homework4ReportMaker(df, predictors, response)
    hw4_html = hw4_report_maker.make_plots_rankings()
    midterm_report_maker = MidtermReportMaker(df, predictors, response)
    midterm_html = midterm_report_maker.make_correlations_bruteforce()
    models = [tree.DecisionTreeClassifier(), svm.SVC()]
    model_html = models_test(df, predictors, response, models)
    complete_html = hw4_html + midterm_html + model_html

    with open("homework5/report.html", "w+") as file:
        file.write(complete_html)
    file.close()
    filename = f"file:///{os.getcwd()}/homework5/report.html"
    webbrowser.open_new_tab(filename)


if __name__ == "__main__":
    sys.exit(main())
