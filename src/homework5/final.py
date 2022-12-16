import os
import sys
import webbrowser

import pandas as pd
import sqlalchemy
from homework4_main import Homework4ReportMaker
from midterm_main import MidtermReportMaker

# from pyspark import StorageLevel
# from pyspark.sql import SparkSession
from sklearn import svm, tree
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def models_test(df, predictors, response, models):
    X = df[predictors]
    Y = df[response]
    output = "\n\n<h2>Model Scores</h2>"
    best_name = ""
    best_score = 0

    sizes = [0.2, 0.4, 0.6, 0.8]

    for i in range(len(models)):
        for size in sizes:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                Y,
                test_size=size,
                shuffle=False,
                random_state=None,
            )

            model = models[i]
            model = model.fit(X_train, y_train)
            model_name = str(model)
            model_score = model.score(X_test, y_test)

            y_pred = model.predict(X_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            matthews = matthews_corrcoef(y_test, y_pred)

            if model_score > best_score:
                best_score = model_score
                best_name = model_name

            output += f"\n<h3>Test_size: {size}</h3>"
            output += (
                f"\n<h3>* {model_name} Score: {model_score} Precision: {precision} Recall: {recall} Accuracy: "
                f"{accuracy} F1: {f1} Matthews: {matthews}</h3>"
            )

    output += (
        f"\n\n<h3>The best model tested was {best_name} (Score = {best_score})</h3>"
    )

    return output


def main():
    '''
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    database = "baseball"
    user = "root"
    password = ""
    server = "localhost"
    port = 3306

    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    sql_query = """SELECT * FROM final_features"""

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
    '''

    """
    user = "root"
    password = "password123"  # pragma: allowlist secret
    host = "mariadb:3306"
    db = "baseball"
    connection = f"mariadb-mariadbconnector://{user}:{password}@{host}/{db}"
    engine = sqlalchemy.create_engine(connection)
    """

    db_user = "root"
    db_pass = "password123"  # pragma: allowlist secret
    db_host = "mariadb-nmo:3306"
    db_database = "baseball"
    connect_string = f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"  # pragma
    sql_engine = sqlalchemy.create_engine(connect_string)
    query = """SELECT * FROM final_features"""
    df = pd.read_sql_query(query, sql_engine)

    # sql_query = "SELECT * FROM final_features"
    # df = pd.read_sql_query(sql_query, engine)

    # home batters 107
    df["hba107_doubles"] = df["hba107_doubles"].astype("float")
    df["hba107_triples"] = df["hba107_triples"].astype("float")
    df["hba107_HR"] = df["hba107_HR"].astype("float")
    df["hba107_BB"] = df["hba107_BB"].astype("float")
    df["hba107_K"] = df["hba107_K"].astype("float")
    df["hba107_R"] = df["hba107_R"].astype("float")
    df["hba107_wOBA"] = df["hba107_wOBA"].astype("float")
    df["hba107_BA"] = df["hba107_BA"].astype("float")
    df["hba107_OBP"] = df["hba107_OBP"].astype("float")
    df["hba107_SLG"] = df["hba107_SLG"].astype("float")
    df["hba107_OPS"] = df["hba107_OPS"].astype("float")
    df["h107_RDIFF"] = df["h107_RDIFF"].astype("float")

    # home starters 107
    df["hsp107_BB"] = df["hsp107_BB"].astype("float")
    df["hsp107_K"] = df["hsp107_K"].astype("float")
    df["hsp107_HR"] = df["hsp107_HR"].astype("float")
    df["hsp107_WHIP"] = df["hsp107_WHIP"].astype("float")
    df["hsp107_BAA"] = df["hsp107_BAA"].astype("float")
    df["hsp107_FIP"] = df["hsp107_FIP"].astype("float")
    df["hsp107_KBB"] = df["hsp107_KBB"].astype("float")
    df["hsp107_IP"] = df["hsp107_IP"].astype("float")
    df["hsp107_PT"] = df["hsp107_PT"].astype("float")
    df["h107_RA"] = df["h107_RA"].astype("float")

    # home bullpen 107
    df["hbp107_BB"] = df["hbp107_BB"].astype("float")
    df["hbp107_K"] = df["hbp107_K"].astype("float")
    df["hbp107_HR"] = df["hbp107_HR"].astype("float")
    df["hbp107_WHIP"] = df["hbp107_WHIP"].astype("float")
    df["hbp107_BAA"] = df["hbp107_BAA"].astype("float")
    df["hbp107_FIP"] = df["hbp107_FIP"].astype("float")
    df["hbp107_KBB"] = df["hbp107_KBB"].astype("float")
    df["hbp107_IP"] = df["hbp107_IP"].astype("float")
    df["hbp107_PT"] = df["hbp107_PT"].astype("float")

    # away batters 107
    df["aba107_doubles"] = df["aba107_doubles"].astype("float")
    df["aba107_triples"] = df["aba107_triples"].astype("float")
    df["aba107_HR"] = df["aba107_HR"].astype("float")
    df["aba107_BB"] = df["aba107_BB"].astype("float")
    df["aba107_K"] = df["aba107_K"].astype("float")
    df["aba107_R"] = df["aba107_R"].astype("float")
    df["aba107_wOBA"] = df["aba107_wOBA"].astype("float")
    df["aba107_BA"] = df["aba107_BA"].astype("float")
    df["aba107_OBP"] = df["aba107_OBP"].astype("float")
    df["aba107_SLG"] = df["aba107_SLG"].astype("float")
    df["aba107_OPS"] = df["aba107_OPS"].astype("float")
    df["a107_RDIFF"] = df["a107_RDIFF"].astype("float")

    # away starters 107
    df["asp107_BB"] = df["asp107_BB"].astype("float")
    df["asp107_K"] = df["asp107_K"].astype("float")
    df["asp107_HR"] = df["asp107_HR"].astype("float")
    df["asp107_WHIP"] = df["asp107_WHIP"].astype("float")
    df["asp107_BAA"] = df["asp107_BAA"].astype("float")
    df["asp107_FIP"] = df["asp107_FIP"].astype("float")
    df["asp107_KBB"] = df["asp107_KBB"].astype("float")
    df["asp107_IP"] = df["asp107_IP"].astype("float")
    df["asp107_PT"] = df["asp107_PT"].astype("float")
    df["a107_RA"] = df["a107_RA"].astype("float")

    # away bullpen 107
    df["abp107_BB"] = df["abp107_BB"].astype("float")
    df["abp107_K"] = df["abp107_K"].astype("float")
    df["abp107_HR"] = df["abp107_HR"].astype("float")
    df["abp107_WHIP"] = df["abp107_WHIP"].astype("float")
    df["abp107_BAA"] = df["abp107_BAA"].astype("float")
    df["abp107_FIP"] = df["abp107_FIP"].astype("float")
    df["abp107_KBB"] = df["abp107_KBB"].astype("float")
    df["abp107_IP"] = df["abp107_IP"].astype("float")
    df["abp107_PT"] = df["abp107_PT"].astype("float")

    # diff batters 107
    df["dba107_doubles"] = df["dba107_doubles"].astype("float")
    df["dba107_triples"] = df["dba107_triples"].astype("float")
    df["dba107_HR"] = df["dba107_HR"].astype("float")
    df["dba107_BB"] = df["dba107_BB"].astype("float")
    df["dba107_K"] = df["dba107_K"].astype("float")
    df["dba107_R"] = df["dba107_R"].astype("float")
    df["dba107_wOBA"] = df["dba107_wOBA"].astype("float")
    df["dba107_BA"] = df["dba107_BA"].astype("float")
    df["dba107_OBP"] = df["dba107_OBP"].astype("float")
    df["dba107_SLG"] = df["dba107_SLG"].astype("float")
    df["dba107_OPS"] = df["dba107_OPS"].astype("float")
    df["d107_RDIFF"] = df["d107_RDIFF"].astype("float")

    # diff starters 107
    df["dsp107_BB"] = df["dsp107_BB"].astype("float")
    df["dsp107_K"] = df["dsp107_K"].astype("float")
    df["dsp107_HR"] = df["dsp107_HR"].astype("float")
    df["dsp107_WHIP"] = df["dsp107_WHIP"].astype("float")
    df["dsp107_BAA"] = df["dsp107_BAA"].astype("float")
    df["dsp107_FIP"] = df["dsp107_FIP"].astype("float")
    df["dsp107_KBB"] = df["dsp107_KBB"].astype("float")
    df["dsp107_IP"] = df["dsp107_IP"].astype("float")
    df["dsp107_PT"] = df["dsp107_PT"].astype("float")
    df["d107_RA"] = df["d107_RA"].astype("float")

    # diff bullpen 107
    df["dbp107_BB"] = df["dbp107_BB"].astype("float")
    df["dbp107_K"] = df["dbp107_K"].astype("float")
    df["dbp107_HR"] = df["dbp107_HR"].astype("float")
    df["dbp107_WHIP"] = df["dbp107_WHIP"].astype("float")
    df["dbp107_BAA"] = df["dbp107_BAA"].astype("float")
    df["dbp107_FIP"] = df["dbp107_FIP"].astype("float")
    df["dbp107_KBB"] = df["dbp107_KBB"].astype("float")
    df["dbp107_IP"] = df["dbp107_IP"].astype("float")
    df["dbp107_PT"] = df["dbp107_PT"].astype("float")

    predictors = [
        "hba107_doubles",
        "hba107_triples",
        "hba107_HR",
        "hba107_BB",
        "hba107_K",
        "hba107_R",
        "hba107_wOBA",
        "hba107_BA",
        "hba107_OBP",
        "hba107_SLG",
        "hba107_OPS",
        "h107_RDIFF",
        "hsp107_BB",
        "hsp107_K",
        "hsp107_HR",
        "hsp107_WHIP",
        "hsp107_BAA",
        "hsp107_FIP",
        "hsp107_KBB",
        "hsp107_IP",
        "hsp107_PT",
        "h107_RA",
        "hbp107_BB",
        "hbp107_K",
        "hbp107_HR",
        "hbp107_WHIP",
        "hbp107_BAA",
        "hbp107_FIP",
        "hbp107_KBB",
        "hbp107_IP",
        "hbp107_PT",
        "aba107_doubles",
        "aba107_triples",
        "aba107_HR",
        "aba107_BB",
        "aba107_K",
        "aba107_R",
        "aba107_wOBA",
        "aba107_BA",
        "aba107_OBP",
        "aba107_SLG",
        "aba107_OPS",
        "a107_RDIFF",
        "asp107_BB",
        "asp107_K",
        "asp107_HR",
        "asp107_WHIP",
        "asp107_BAA",
        "asp107_FIP",
        "asp107_KBB",
        "asp107_IP",
        "asp107_PT",
        "a107_RA",
        "abp107_BB",
        "abp107_K",
        "abp107_HR",
        "abp107_WHIP",
        "abp107_BAA",
        "abp107_FIP",
        "abp107_KBB",
        "abp107_IP",
        "abp107_PT",
        "dba107_doubles",
        "dba107_triples",
        "dba107_HR",
        "dba107_BB",
        "dba107_K",
        "dba107_R",
        "dba107_wOBA",
        "dba107_BA",
        "dba107_OBP",
        "dba107_SLG",
        "dba107_OPS",
        "d107_RDIFF",
        "dsp107_BB",
        "dsp107_K",
        "dsp107_HR",
        "dsp107_WHIP",
        "dsp107_BAA",
        "dsp107_FIP",
        "dsp107_KBB",
        "dsp107_IP",
        "dsp107_PT",
        "d107_RA",
        "dbp107_BB",
        "dbp107_K",
        "dbp107_HR",
        "dbp107_WHIP",
        "dbp107_BAA",
        "dbp107_FIP",
        "dbp107_KBB",
        "dbp107_IP",
        "dbp107_PT",
    ]

    predictors = [
        "hba107_doubles",
        "hba107_HR",
        "hba107_BB",
        "hba107_R",
        "hba107_wOBA",
        "hba107_BA",
        "hba107_OBP",
        "hba107_SLG",
        "hba107_OPS",
        "h107_RDIFF",
        "hsp107_K",
        "hsp107_IP",
        "h107_RA",
        "hbp107_BB",
        "hbp107_K",
        "hbp107_HR",
        "hbp107_WHIP",
        "hbp107_BAA",
        "hbp107_IP",
        "hbp107_PT",
        "aba107_doubles",
        "aba107_triples",
        "aba107_K",
        "a107_RDIFF",
        "asp107_BB",
        "asp107_HR",
        "asp107_WHIP",
        "asp107_BAA",
        "asp107_FIP",
        "a107_RA",
        "abp107_BB",
        "abp107_K",
        "abp107_HR",
        "abp107_IP",
        "abp107_PT",
        "dba107_HR",
        "dba107_BB",
        "dba107_R",
        "dba107_wOBA",
        "dba107_OBP",
        "dba107_SLG",
        "dba107_OPS",
        "d107_RDIFF",
        "dsp107_BB",
        "dsp107_K",
        "dsp107_WHIP",
        "dsp107_BAA",
        "dsp107_FIP",
        "dsp107_IP",
        "dsp107_PT",
        "d107_RA",
        "dbp107_BB",
        "dbp107_K",
        "dbp107_HR",
        "dbp107_WHIP",
        "dbp107_BAA",
        "dbp107_IP",
        "dbp107_PT",
    ]

    predictors = [
        "hba107_HR",
        "hba107_BB",
        "hba107_wOBA",
        "hba107_BA",
        "h107_RDIFF",
        "h107_RA",
        "hbp107_BB",
        "hbp107_HR",
        "hbp107_WHIP",
        "hbp107_PT",
        "aba107_triples",
        "a107_RDIFF",
        "asp107_BB",
        "asp107_HR",
        "asp107_WHIP",
        "asp107_BAA",
        "asp107_FIP",
        "a107_RA",
        "abp107_BB",
        "abp107_HR",
        "abp107_PT",
        "dba107_HR",
        "dba107_BB",
        "dba107_R",
        "dba107_wOBA",
        "d107_RDIFF",
        "dsp107_BB",
        "dsp107_K",
        "dsp107_WHIP",
        "dsp107_BAA",
        "dsp107_FIP",
        "dsp107_IP",
        "dsp107_PT",
        "d107_RA",
        "dbp107_BB",
        "dbp107_K",
        "dbp107_HR",
        "dbp107_WHIP",
        "dbp107_BAA",
    ]

    predictors = [
        "hba107_BB",
        "h107_RDIFF",
        "hbp107_BB",
        "hbp107_HR",
        "hbp107_PT",
        "a107_RDIFF",
        "asp107_WHIP",
        "asp107_BAA",
        "a107_RA",
        "dba107_BB",
        "dba107_R",
        "d107_RDIFF",
        "dsp107_K",
        "dsp107_WHIP",
        "dsp107_BAA",
        "d107_RA",
        "dbp107_BB",
        "dbp107_HR",
    ]

    predictors = [
        "h107_RDIFF",
        "hbp107_BB",
        "hbp107_HR",
        "hbp107_PT",
        "a107_RDIFF",
        "dba107_R",
        "d107_RDIFF",
        "dsp107_K",
        "dbp107_BB",
        "dbp107_HR",
    ]

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
        AdaBoostClassifier(random_state=123),
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
