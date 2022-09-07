import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def main():
    df = pd.read_csv(
        "../src/iris.data",
        names=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
            "class",
        ],
    )

    for i in df.columns[:4]:
        print(i)
        print("Mean: " + str(np.mean(df[i])))
        print("Min: " + str(np.min(df[i])))
        print("Max: " + str(np.max(df[i])))
        print("Q1: " + str(np.quantile(df[i], 0.25)))
        print("Q2: " + str(np.quantile(df[i], 0.50)))
        print("Q3: " + str(np.quantile(df[i], 0.75)))
        print()

    fig = px.scatter(
        df,
        title="Scatter plot - Sepal Width vs Sepal Length",
        x="sepal length (cm)",
        y="sepal width (cm)",
        color="class",
    )
    fig.show()

    fig = px.violin(
        df, title="Violin plot - Sepal Width", x="class", y="sepal width (cm)"
    )
    fig.show()

    fig = px.box(df, title="Box plot - Sepal Length", x="class", y="sepal length (cm)")
    fig.show()

    fig = px.scatter(
        df,
        title="Scatter plot - Petal Width vs Petal Length",
        x="petal length (cm)",
        y="petal width (cm)",
        color="class",
    )
    fig.show()

    fig = px.violin(
        df, title="Violin plot - Petal Width", x="class", y="petal width (cm)"
    )
    fig.show()

    fig = px.box(df, title="Box plot - Petal Length", x="class", y="petal length (cm)")
    fig.show()

    X_orig = df[
        [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
    ].values

    y = df["class"].values

    pipeline_RandomForest = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )

    pipeline_RandomForest.fit(X_orig, y)

    probability = pipeline_RandomForest.predict_proba(X_orig)
    prediction = pipeline_RandomForest.predict(X_orig)
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")

    pipeline_Neighbors = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("KNeighbors", KNeighborsClassifier(n_neighbors=3)),
        ]
    )

    pipeline_Neighbors.fit(X_orig, y)

    probability = pipeline_Neighbors.predict_proba(X_orig)
    prediction = pipeline_Neighbors.predict(X_orig)
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")

    pipeline_DecisionTree = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("DecisionTree", DecisionTreeClassifier(random_state=1234)),
        ]
    )

    pipeline_DecisionTree.fit(X_orig, y)

    probability = pipeline_DecisionTree.predict_proba(X_orig)
    prediction = pipeline_DecisionTree.predict(X_orig)
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")


if __name__ == "__main__":
    sys.exit(main())
