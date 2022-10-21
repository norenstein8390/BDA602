import random
import sys
from typing import List

import pandas as pd
import seaborn
from scipy.stats import pearsonr
from sklearn import datasets

TITANIC_PREDICTORS = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "embarked",
    "parch",
    "fare",
    "who",
    "adult_male",
    "deck",
    "embark_town",
    "alone",
    "class",
]


def get_test_data_set(data_set_name: str = None) -> (pd.DataFrame, List[str], str):
    """Function to load a few test data sets

    :param:
    data_set_name : string, optional
        Data set to load

    :return:
    data_set : :class:`pd.DataFrame`
        Tabular data, possibly with some preprocessing applied.
    predictors :list[str]
        List of predictor variables
    response: str
        Response variable
    """
    seaborn_data_sets = ["mpg", "tips", "titanic", "titanic_2"]
    sklearn_data_sets = ["boston", "diabetes", "breast_cancer"]
    all_data_sets = seaborn_data_sets + sklearn_data_sets

    if data_set_name is None:
        data_set_name = random.choice(all_data_sets)
    else:
        if data_set_name not in all_data_sets:
            raise Exception(f"Data set choice not valid: {data_set_name}")

    if data_set_name in seaborn_data_sets:
        if data_set_name == "mpg":
            data_set = seaborn.load_dataset(name="mpg").dropna().reset_index()
            predictors = [
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "origin",
                "name",
            ]
            response = "mpg"
        elif data_set_name == "tips":
            data_set = seaborn.load_dataset(name="tips").dropna().reset_index()
            predictors = [
                "total_bill",
                "sex",
                "smoker",
                "day",
                "time",
                "size",
            ]
            response = "tip"
        elif data_set_name == "titanic":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "survived"
        elif data_set_name == "titanic_2":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "alive"
    elif data_set_name in sklearn_data_sets:
        if data_set_name == "boston":
            data = datasets.load_boston()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)
            data_set["CHAS"] = data_set["CHAS"].astype(str)
        elif data_set_name == "diabetes":
            data = datasets.load_diabetes()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)
            data_set["gender"] = ["1" if i > 0 else "0" for i in data_set["sex"]]
        elif data_set_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)

        data_set["target"] = data.target
        predictors = data.feature_names
        response = "target"

    print(f"Data set selected: {data_set_name}")
    return data_set, predictors, response


def split_dataset(df, predictors):
    cat_predictors = []
    cont_predictors = []

    for predictor in predictors:
        if df[predictor].dtype == float or df[predictor].dtype == int:
            cont_predictors.append(predictor)
        else:
            cat_predictors.append(predictor)

    print(cont_predictors)

    return cat_predictors + cont_predictors, len(cat_predictors)


def cont_cont_correlation(df, predictors, cont_predictor_start):
    cont_predictors = predictors[cont_predictor_start:]

    for i in range(len(cont_predictors)):
        x_predictor = cont_predictors[i]

        for j in range(i + 1, len(cont_predictors)):
            y_predictor = cont_predictors[j]
            print(x_predictor + " and " + y_predictor)
            print(pearsonr(df[x_predictor], df[y_predictor])[0])
            print(abs(pearsonr(df[x_predictor], df[y_predictor])[0]))


def main():
    # Step 1 - Given DataFrame, predictors, and response
    test_data_set = get_test_data_set("titanic")
    df = test_data_set[0]
    predictors = test_data_set[1]
    # response = test_data_set[2]

    # Step 2 - Split dataset
    predictors, cont_predictor_start = split_dataset(df, predictors)
    print(predictors[cont_predictor_start:])

    # Step 3 - Correlation metrics cont/cont pairs
    cont_cont_correlation(df, predictors, cont_predictor_start)


if __name__ == "__main__":
    sys.exit(main())
