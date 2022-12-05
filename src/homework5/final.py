import os
import sys
import webbrowser

from homework4_main import Homework4ReportMaker
from midterm_main import MidtermReportMaker
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from sklearn import svm, tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def models_test(df, predictors, response, models):
    X = df[predictors]
    Y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False, random_state=None
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

    sql_query = """SELECT * FROM final"""

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

    df["home_batters_H9"] = df["home_batters_H9"].astype("float")
    df["away_batters_H9"] = df["away_batters_H9"].astype("float")
    df["home_batters_doubles9"] = df["home_batters_doubles9"].astype("float")
    df["away_batters_doubles9"] = df["away_batters_doubles9"].astype("float")
    df["home_batters_triples9"] = df["home_batters_triples9"].astype("float")
    df["away_batters_triples9"] = df["away_batters_triples9"].astype("float")
    df["home_batters_HR9"] = df["home_batters_HR9"].astype("float")
    df["away_batters_HR9"] = df["away_batters_HR9"].astype("float")
    df["home_batters_BB9"] = df["home_batters_BB9"].astype("float")
    df["away_batters_BB9"] = df["away_batters_BB9"].astype("float")
    df["home_batters_K9"] = df["home_batters_K9"].astype("float")
    df["away_batters_K9"] = df["away_batters_K9"].astype("float")
    df["home_batters_R9"] = df["home_batters_R9"].astype("float")
    df["away_batters_R9"] = df["away_batters_R9"].astype("float")
    df["home_RDIFF9"] = df["home_RDIFF9"].astype("float")
    df["away_RDIFF9"] = df["away_RDIFF9"].astype("float")
    df["home_BA"] = df["home_BA"].astype("float")
    df["away_BA"] = df["away_BA"].astype("float")
    df["home_OBP"] = df["home_OBP"].astype("float")
    df["away_OBP"] = df["away_OBP"].astype("float")
    df["home_SLG"] = df["home_SLG"].astype("float")
    df["away_SLG"] = df["away_SLG"].astype("float")
    df["home_OPS"] = df["home_OPS"].astype("float")
    df["away_OPS"] = df["away_OPS"].astype("float")
    df["home_wOBA"] = df["home_wOBA"].astype("float")
    df["away_wOBA"] = df["away_wOBA"].astype("float")
    df["home_pitchers_H9"] = df["home_pitchers_H9"].astype("float")
    df["away_pitchers_H9"] = df["away_pitchers_H9"].astype("float")
    df["home_pitchers_R9"] = df["home_pitchers_R9"].astype("float")
    df["away_pitchers_R9"] = df["away_pitchers_R9"].astype("float")
    df["home_pitchers_BB9"] = df["home_pitchers_BB9"].astype("float")
    df["away_pitchers_BB9"] = df["away_pitchers_BB9"].astype("float")
    df["home_pitchers_K9"] = df["home_pitchers_K9"].astype("float")
    df["away_pitchers_K9"] = df["away_pitchers_K9"].astype("float")
    df["home_pitchers_HR9"] = df["home_pitchers_HR9"].astype("float")
    df["away_pitchers_HR9"] = df["away_pitchers_HR9"].astype("float")
    df["home_WHIP"] = df["home_WHIP"].astype("float")
    df["away_WHIP"] = df["away_WHIP"].astype("float")
    df["home_BAA"] = df["home_BAA"].astype("float")
    df["away_BAA"] = df["away_BAA"].astype("float")
    df["home_FIP"] = df["home_FIP"].astype("float")
    df["away_FIP"] = df["away_FIP"].astype("float")
    df["home_batters10_H9"] = df["home_batters10_H9"].astype("float")
    df["away_batters10_H9"] = df["away_batters10_H9"].astype("float")
    df["home_batters10_doubles9"] = df["home_batters10_doubles9"].astype("float")
    df["away_batters10_doubles9"] = df["away_batters10_doubles9"].astype("float")
    df["home_batters10_triples9"] = df["home_batters10_triples9"].astype("float")
    df["away_batters10_triples9"] = df["away_batters10_triples9"].astype("float")
    df["home_batters10_HR9"] = df["home_batters10_HR9"].astype("float")
    df["away_batters10_HR9"] = df["away_batters10_HR9"].astype("float")
    df["home_batters10_BB9"] = df["home_batters10_BB9"].astype("float")
    df["away_batters10_BB9"] = df["away_batters10_BB9"].astype("float")
    df["home_batters10_K9"] = df["home_batters10_K9"].astype("float")
    df["away_batters10_K9"] = df["away_batters10_K9"].astype("float")
    df["home_batters10_R9"] = df["home_batters10_R9"].astype("float")
    df["away_batters10_R9"] = df["away_batters10_R9"].astype("float")
    df["home10_RDIFF9"] = df["home10_RDIFF9"].astype("float")
    df["away10_RDIFF9"] = df["away10_RDIFF9"].astype("float")
    df["home10_BA"] = df["home10_BA"].astype("float")
    df["away10_BA"] = df["away10_BA"].astype("float")
    df["home10_OBP"] = df["home10_OBP"].astype("float")
    df["away10_OBP"] = df["away10_OBP"].astype("float")
    df["home10_SLG"] = df["home10_SLG"].astype("float")
    df["away10_SLG"] = df["away10_SLG"].astype("float")
    df["home10_OPS"] = df["home10_OPS"].astype("float")
    df["away10_OPS"] = df["away10_OPS"].astype("float")
    df["home10_wOBA"] = df["home10_wOBA"].astype("float")
    df["away10_wOBA"] = df["away10_wOBA"].astype("float")
    df["home_pitchers10_H9"] = df["home_pitchers10_H9"].astype("float")
    df["away_pitchers10_H9"] = df["away_pitchers10_H9"].astype("float")
    df["home_pitchers10_R9"] = df["home_pitchers10_R9"].astype("float")
    df["away_pitchers10_R9"] = df["away_pitchers10_R9"].astype("float")
    df["home_pitchers10_BB9"] = df["home_pitchers10_BB9"].astype("float")
    df["away_pitchers10_BB9"] = df["away_pitchers10_BB9"].astype("float")
    df["home_pitchers10_K9"] = df["home_pitchers10_K9"].astype("float")
    df["away_pitchers10_K9"] = df["away_pitchers10_K9"].astype("float")
    df["home_pitchers10_HR9"] = df["home_pitchers10_HR9"].astype("float")
    df["away_pitchers10_HR9"] = df["away_pitchers10_HR9"].astype("float")
    df["home10_WHIP"] = df["home10_WHIP"].astype("float")
    df["away10_WHIP"] = df["away10_WHIP"].astype("float")
    df["home10_BAA"] = df["home10_BAA"].astype("float")
    df["away10_BAA"] = df["away10_BAA"].astype("float")
    df["home10_FIP"] = df["home10_FIP"].astype("float")
    df["away10_FIP"] = df["away10_FIP"].astype("float")

    predictors = [
        "home_batters_H9",
        "away_batters_H9",
        "home_batters_doubles9",
        "away_batters_doubles9",
        "home_batters_triples9",
        "away_batters_triples9",
        "home_batters_HR9",
        "away_batters_HR9",
        "home_batters_BB9",
        "away_batters_BB9",
        "home_batters_K9",
        "away_batters_K9",
        "home_batters_R9",
        "away_batters_R9",
        "home_RDIFF9",
        "away_RDIFF9",
        "home_BA",
        "away_BA",
        "home_OBP",
        "away_OBP",
        "home_SLG",
        "away_SLG",
        "home_OPS",
        "away_OPS",
        "home_wOBA",
        "away_wOBA",
        "home_pitchers_H9",
        "away_pitchers_H9",
        "home_pitchers_R9",
        "away_pitchers_R9",
        "home_pitchers_BB9",
        "away_pitchers_BB9",
        "home_pitchers_K9",
        "away_pitchers_K9",
        "home_pitchers_HR9",
        "away_pitchers_HR9",
        "home_WHIP",
        "away_WHIP",
        "home_BAA",
        "away_BAA",
        "home_FIP",
        "away_FIP",
        "home_batters10_H9",
        "away_batters10_H9",
        "home_batters10_doubles9",
        "away_batters10_doubles9",
        "home_batters10_triples9",
        "away_batters10_triples9",
        "home_batters10_HR9",
        "away_batters10_HR9",
        "home_batters10_BB9",
        "away_batters10_BB9",
        "home_batters10_K9",
        "away_batters10_K9",
        "home_batters10_R9",
        "away_batters10_R9",
        "home10_RDIFF9",
        "away10_RDIFF9",
        "home10_BA",
        "away10_BA",
        "home10_OBP",
        "away10_OBP",
        "home10_SLG",
        "away10_SLG",
        "home10_OPS",
        "away10_OPS",
        "home10_wOBA",
        "away10_wOBA",
        "home_pitchers10_H9",
        "away_pitchers10_H9",
        "home_pitchers10_R9",
        "away_pitchers10_R9",
        "home_pitchers10_BB9",
        "away_pitchers10_BB9",
        "home_pitchers10_K9",
        "away_pitchers10_K9",
        "home_pitchers10_HR9",
        "away_pitchers10_HR9",
        "home10_WHIP",
        "away10_WHIP",
        "home10_BAA",
        "away10_BAA",
        "home10_FIP",
        "away10_FIP",
    ]
    """

    df["diff_batters_H9"] = df["diff_batters_H9"].astype("float")
    df["diff_batters_doubles9"] = df["diff_batters_doubles9"].astype("float")
    df["diff_batters_triples9"] = df["diff_batters_triples9"].astype("float")
    df["diff_batters_HR9"] = df["diff_batters_HR9"].astype("float")
    df["diff_batters_BB9"] = df["diff_batters_BB9"].astype("float")
    df["diff_batters_K9"] = df["diff_batters_K9"].astype("float")
    df["diff_batters_R9"] = df["diff_batters_R9"].astype("float")
    df["diff_RDIFF9"] = df["diff_RDIFF9"].astype("float")
    df["diff_BA"] = df["diff_BA"].astype("float")
    df["diff_OBP"] = df["diff_OBP"].astype("float")
    df["diff_SLG"] = df["diff_SLG"].astype("float")
    df["diff_OPS"] = df["diff_OPS"].astype("float")
    df["diff_wOBA"] = df["diff_wOBA"].astype("float")
    df["diff_pitchers_H9"] = df["diff_pitchers_H9"].astype("float")
    df["diff_pitchers_R9"] = df["diff_pitchers_R9"].astype("float")
    df["diff_pitchers_BB9"] = df["diff_pitchers_BB9"].astype("float")
    df["diff_pitchers_K9"] = df["diff_pitchers_K9"].astype("float")
    df["diff_pitchers_HR9"] = df["diff_pitchers_HR9"].astype("float")
    df["diff_WHIP"] = df["diff_WHIP"].astype("float")
    df["diff_BAA"] = df["diff_BAA"].astype("float")
    df["diff_FIP"] = df["diff_FIP"].astype("float")
    df["diff_batters10_H9"] = df["diff_batters10_H9"].astype("float")
    df["diff_batters10_doubles9"] = df["diff_batters10_doubles9"].astype("float")
    df["diff_batters10_triples9"] = df["diff_batters10_triples9"].astype("float")
    df["diff_batters10_HR9"] = df["diff_batters10_HR9"].astype("float")
    df["diff_batters10_BB9"] = df["diff_batters10_BB9"].astype("float")
    df["diff_batters10_K9"] = df["diff_batters10_K9"].astype("float")
    df["diff_batters10_R9"] = df["diff_batters10_R9"].astype("float")
    df["diff10_RDIFF9"] = df["diff10_RDIFF9"].astype("float")
    df["diff10_BA"] = df["diff10_BA"].astype("float")
    df["diff10_OBP"] = df["diff10_OBP"].astype("float")
    df["diff10_SLG"] = df["diff10_SLG"].astype("float")
    df["diff10_OPS"] = df["diff10_OPS"].astype("float")
    df["diff10_wOBA"] = df["diff10_wOBA"].astype("float")
    df["diff_pitchers10_H9"] = df["diff_pitchers10_H9"].astype("float")
    df["diff_pitchers10_R9"] = df["diff_pitchers10_R9"].astype("float")
    df["diff_pitchers10_BB9"] = df["diff_pitchers10_BB9"].astype("float")
    df["diff_pitchers10_K9"] = df["diff_pitchers10_K9"].astype("float")
    df["diff_pitchers10_HR9"] = df["diff_pitchers10_HR9"].astype("float")
    df["diff10_WHIP"] = df["diff10_WHIP"].astype("float")
    df["diff10_BAA"] = df["diff10_BAA"].astype("float")
    df["diff10_FIP"] = df["diff10_FIP"].astype("float")

    predictors = [
        "diff_RDIFF9",
        "diff_FIP",
        "diff_wOBA",
        "diff10_RDIFF9",
        "diff10_FIP",
        "diff10_wOBA"
    ]
    """

    response = "HomeTeamWins"

    hw4_report_maker = Homework4ReportMaker(df, predictors, response)
    hw4_html = hw4_report_maker.make_plots_rankings()
    midterm_report_maker = MidtermReportMaker(df, predictors, response)
    midterm_html = midterm_report_maker.make_correlations_bruteforce()
    models = [
        tree.DecisionTreeClassifier(random_state=123),
        svm.SVC(random_state=123),
        RandomForestClassifier(random_state=123),
        LogisticRegression(random_state=123),
        GaussianNB(),
        KNeighborsClassifier(),
        GradientBoostingClassifier(random_state=123),
        SGDClassifier(random_state=123),
    ]
    model_html = models_test(df, predictors, response, models)
    complete_html = hw4_html + midterm_html + model_html

    with open("homework5/report.html", "w+") as file:
        file.write(complete_html)
    file.close()
    filename = f"file:///{os.getcwd()}/homework5/report.html"
    webbrowser.open_new_tab(filename)


if __name__ == "__main__":
    sys.exit(main())
