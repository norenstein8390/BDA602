import random
import sys
import warnings
from typing import List

import numpy as np
import pandas as pd
import seaborn
from plotly import express as px
from scipy import stats
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

    # print(cat_predictors)
    # print(cont_predictors)

    return cat_predictors + cont_predictors, len(cat_predictors)


def cont_cont_correlation(df, predictors, cont_predictor_start, data):
    cont_predictors = predictors[cont_predictor_start:]

    for i in range(len(cont_predictors)):
        x_predictor = cont_predictors[i]

        for j in range(i + 1, len(cont_predictors)):
            y_predictor = cont_predictors[j]
            data["Predictors"].append(x_predictor + " and " + y_predictor)
            # print(x_predictor + " and " + y_predictor)
            data["Pearson's r"].append(pearsonr(df[x_predictor], df[y_predictor])[0])
            # print(pearsonr(df[x_predictor], df[y_predictor])[0])
            data["Absolute Value of Correlation"].append(
                abs(pearsonr(df[x_predictor], df[y_predictor])[0])
            )
            # print(abs(pearsonr(df[x_predictor], df[y_predictor])[0]))

            fig = px.scatter(df, x=x_predictor, y=y_predictor, trendline="ols")
            fig.update_layout(
                title="{} vs. {}: Scatter Plot".format(x_predictor, y_predictor),
                xaxis_title=x_predictor,
                yaxis_title=y_predictor,
            )
            fig.show()
            # fig.write_html("output/figs/" + predictor + "/scatter_plot.html")


def cat_cont_correlation_ratio(categories, values):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    f_cat, _ = pd.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def cont_cat_correlation(df, predictors, cont_predictor_start, data):
    cont_predictors = predictors[cont_predictor_start:]
    cat_predictors = predictors[:cont_predictor_start]

    for i in range(len(cont_predictors)):
        cont_predictor = cont_predictors[i]

        for j in range(len(cat_predictors)):
            cat_predictor = cat_predictors[j]
            data["Predictors"].append(cont_predictor + " and " + cat_predictor)
            # print(cont_predictor + " and " + cat_predictor)
            data["Correlation Ratio"].append(
                cat_cont_correlation_ratio(
                    np.array(df[cat_predictor]), np.array(df[cont_predictor])
                )
            )
            # print(cat_cont_correlation_ratio(np.array(df[cat_predictor]), np.array(df[cont_predictor])))
            data["Absolute Value of Correlation"].append(
                abs(
                    cat_cont_correlation_ratio(
                        np.array(df[cat_predictor]), np.array(df[cont_predictor])
                    )
                )
            )
            # print(abs(cat_cont_correlation_ratio(np.array(df[cat_predictor]), np.array(df[cont_predictor]))))


def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    """
    Calculates correlation statistic for categorical-categorical association.
    The two measures supported are:
    1. Cramer'V ( default )
    2. Tschuprow'T

    SOURCES:
    1.) CODE: https://github.com/MavericksDS/pycorr
    2.) Used logic from:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
        to ignore yates correction factor on 2x2
    3.) Haven't validated Tschuprow

    Bias correction and formula's taken from :
    https://www.researchgate.net/publication/270277061_A_bias-correction_for_Cramer's_V_and_Tschuprow's_T

    Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
    Parameters:
    -----------
    x : list / ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
    tschuprow : Boolean, default = False
               For choosing Tschuprow as measure
    Returns:
    --------
    float in the range of [0,1]
    """
    corr_coeff = np.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def cat_cat_correlation(df, predictors, cont_predictor_start, data):
    cat_predictors = predictors[:cont_predictor_start]

    for i in range(len(cat_predictors)):
        x_predictor = cat_predictors[i]

        for j in range(i + 1, len(cat_predictors)):
            y_predictor = cat_predictors[j]
            data["Predictors"].append(x_predictor + " and " + y_predictor)
            # print(x_predictor + " and " + y_predictor)
            data["Cramer's V"].append(cat_correlation(df[x_predictor], df[y_predictor]))
            # print(cat_correlation(df[x_predictor], df[y_predictor]))
            data["Absolute Value of Correlation"].append(
                abs(cat_correlation(df[x_predictor], df[y_predictor]))
            )
            # print(abs(cat_correlation(df[x_predictor], df[y_predictor])))


def main():
    # Step 1 - Given DataFrame, predictors, and response
    test_data_set = get_test_data_set("titanic")
    df = test_data_set[0]
    predictors = test_data_set[1]
    # response = test_data_set[2]

    # Step 2 - Split dataset
    predictors, cont_predictor_start = split_dataset(df, predictors)

    # Step 3+6 - Correlation metrics cont/cont pairs + put in table
    cont_cont_correlation_data = {
        "Predictors": [],
        "Pearson's r": [],
        "Absolute Value of Correlation": [],
    }
    cont_cont_correlation(
        df, predictors, cont_predictor_start, cont_cont_correlation_data
    )
    cont_cont_correlation_df = pd.DataFrame(cont_cont_correlation_data)
    cont_cont_correlation_df = cont_cont_correlation_df.sort_values(
        by=["Absolute Value of Correlation"], ascending=False
    )

    # Step 4+6 - Correlation metrics cont/cat pairs + put in table
    cont_cat_correlation_data = {
        "Predictors": [],
        "Correlation Ratio": [],
        "Absolute Value of Correlation": [],
    }
    cont_cat_correlation(
        df, predictors, cont_predictor_start, cont_cat_correlation_data
    )
    cont_cat_correlation_df = pd.DataFrame(cont_cat_correlation_data)
    cont_cat_correlation_df = cont_cat_correlation_df.sort_values(
        by=["Absolute Value of Correlation"], ascending=False
    )

    # Step 5+6 - Correlation metrics  cat/cat pairs + put in table
    cat_cat_correlation_data = {
        "Predictors": [],
        "Cramer's V": [],
        "Absolute Value of Correlation": [],
    }
    cat_cat_correlation(df, predictors, cont_predictor_start, cat_cat_correlation_data)
    cat_cat_correlation_df = pd.DataFrame(cat_cat_correlation_data)
    cat_cat_correlation_df = cat_cat_correlation_df.sort_values(
        by=["Absolute Value of Correlation"], ascending=False
    )


if __name__ == "__main__":
    sys.exit(main())
