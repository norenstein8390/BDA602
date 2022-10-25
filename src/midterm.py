import os
import random
import sys
import warnings
import webbrowser
from typing import List

import numpy as np
import pandas as pd
import seaborn
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
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

    return cat_predictors + cont_predictors, len(cat_predictors)


def cont_cont_plots(df, x_predictor, y_predictor):
    fig = px.scatter(df, x=x_predictor, y=y_predictor, trendline="ols")
    fig.update_layout(
        title="{} vs. {}: Scatter Plot".format(x_predictor, y_predictor),
        xaxis_title=x_predictor,
        yaxis_title=y_predictor,
    )
    name = x_predictor + "_" + y_predictor + "_scatter_plot"
    link = "midterm_output/figs/" + name + ".html"
    fig.write_html(link)
    return name


def cont_cont_correlation(df, predictors, cont_predictor_start, data, response):
    cont_predictors = predictors[cont_predictor_start:]

    for i in range(len(cont_predictors)):
        x_predictor = cont_predictors[i]

        for j in range(i + 1, len(cont_predictors)):
            y_predictor = cont_predictors[j]
            data["Predictors (Cont/Cont)"].append(x_predictor + " and " + y_predictor)
            data["Pearson's r"].append(pearsonr(df[x_predictor], df[y_predictor])[0])
            data["Absolute Value of Correlation"].append(
                abs(pearsonr(df[x_predictor], df[y_predictor])[0])
            )
            data["Scatter Plot"].append(cont_cont_plots(df, x_predictor, y_predictor))

            cont_cont_brute_force(df, x_predictor, y_predictor, response)


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


def cont_cat_plots(df, cat_predictor, cont_predictor):
    # Distribution Plot
    predictors = df[cat_predictor].unique()
    if not df[cat_predictor].dtype == "category":
        predictors.sort()
    hist_data = []
    group_labels = []

    for i in range(len(predictors)):
        predictor_i = predictors[i]
        predictor_i_response = df[df[cat_predictor] == predictor_i][
            cont_predictor
        ].dropna()
        hist_data.append(predictor_i_response)
        group_labels.append("Predictor = {}".format(predictor_i))

    fig1 = ff.create_distplot(hist_data, group_labels, curve_type="normal")
    fig1.update_layout(
        title="{} v. {}: Distribution Plot".format(cat_predictor, cont_predictor),
        xaxis_title=cont_predictor,
        yaxis_title="Distribution",
    )

    # Violin Plot
    fig2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig2.add_trace(
            go.Violin(
                x=np.repeat(curr_group, len(curr_hist)),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )

    fig2.update_layout(
        title="{} vs. {}: Violin Plot".format(cat_predictor, cont_predictor),
        xaxis_title=cat_predictor,
        yaxis_title=cont_predictor,
    )
    # fig1.write_html("output/figs/" + predictor + "/distribution_plot.html")
    # fig2.write_html("output/figs/" + predictor + "/violin_plot.html")

    distribution_name = cont_predictor + "_" + cat_predictor + "_distribution_plot"
    distribution_link = "midterm_output/figs/" + distribution_name + ".html"
    fig1.write_html(distribution_link)
    violin_name = cont_predictor + "_" + cat_predictor + "_violin_plot"
    violin_link = "midterm_output/figs/" + violin_name + ".html"
    fig2.write_html(violin_link)
    return distribution_name, violin_name


def cont_cat_correlation(df, predictors, cont_predictor_start, data, response):
    cont_predictors = predictors[cont_predictor_start:]
    cat_predictors = predictors[:cont_predictor_start]

    for i in range(len(cont_predictors)):
        cont_predictor = cont_predictors[i]

        for j in range(len(cat_predictors)):
            cat_predictor = cat_predictors[j]
            data["Predictors (Cont/Cat)"].append(
                cont_predictor + " and " + cat_predictor
            )
            data["Correlation Ratio"].append(
                cat_cont_correlation_ratio(
                    np.array(df[cat_predictor]), np.array(df[cont_predictor])
                )
            )
            data["Absolute Value of Correlation"].append(
                abs(
                    cat_cont_correlation_ratio(
                        np.array(df[cat_predictor]), np.array(df[cont_predictor])
                    )
                )
            )

            plots = cont_cat_plots(df, cat_predictor, cont_predictor)
            data["Distribution Plot"].append(plots[0])
            data["Violin Plot"].append(plots[1])

            cont_cat_brute_force(df, cont_predictor, cat_predictor, response)


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


def cat_cat_plots(df, x_predictor, y_predictor):
    df.sort_values(by=[x_predictor])

    fig = px.density_heatmap(
        df,
        x=x_predictor,
        y=y_predictor,
        title="{} v. {}: Heatmap".format(x_predictor, y_predictor),
    )

    name = x_predictor + "_" + y_predictor + "_heatmap"
    link = "midterm_output/figs/" + name + ".html"
    fig.write_html(link)
    return name


def cat_cat_correlation(df, predictors, cont_predictor_start, data, response):
    cat_predictors = predictors[:cont_predictor_start]

    for i in range(len(cat_predictors)):
        x_predictor = cat_predictors[i]

        for j in range(i + 1, len(cat_predictors)):
            y_predictor = cat_predictors[j]
            data["Predictors (Cat/Cat)"].append(x_predictor + " and " + y_predictor)
            data["Cramer's V"].append(cat_correlation(df[x_predictor], df[y_predictor]))
            data["Absolute Value of Correlation"].append(
                abs(cat_correlation(df[x_predictor], df[y_predictor]))
            )
            data["Heatmap"].append(cat_cat_plots(df, x_predictor, y_predictor))
            cat_cat_brute_force(df, x_predictor, y_predictor, response)


def cont_cont_matrix(df, predictors, cont_predictor_start, data):
    cont_predictors = predictors[cont_predictor_start:]
    matrix = []

    for i in range(len(cont_predictors)):
        cont_predictor_i = cont_predictors[i]
        row = []

        for j in range(len(cont_predictors)):
            cont_predictor_j = cont_predictors[j]
            row.append(abs(pearsonr(df[cont_predictor_i], df[cont_predictor_j])[0]))

        matrix.append(row)

    print(type(matrix))

    fig = px.imshow(
        matrix,
        labels=dict(
            x="Continuous Predictors",
            y="Continuous Predictors",
            color="Absolute Value of Pearson's r",
        ),
        x=cont_predictors,
        y=cont_predictors,
        title="Continuous vs. Continuous Predictors: Correlation Matrix",
    )

    name = "cont_cont_correlation_matrix"
    link = "midterm_output/figs/" + name + ".html"
    fig.write_html(link)
    data["Cont/Cont Correlation Matrix"].append(name)


def cont_cat_matrix(df, predictors, cont_predictor_start, data):
    cont_predictors = predictors[cont_predictor_start:]
    cat_predictors = predictors[:cont_predictor_start]
    matrix = []

    for i in range(len(cont_predictors)):
        cont_predictor = cont_predictors[i]
        row = []

        for j in range(len(cat_predictors)):
            cat_predictor = cat_predictors[j]
            row.append(
                cat_cont_correlation_ratio(
                    np.array(df[cat_predictor]), np.array(df[cont_predictor])
                )
            )

        matrix.append(row)

    fig = px.imshow(
        matrix,
        labels=dict(
            x="Categorical Predictors",
            y="Continuous Predictors",
            color="Correlation Ratio",
        ),
        x=cat_predictors,
        y=cont_predictors,
        title="Categorical vs. Continuous Predictors: Correlation Matrix",
    )

    name = "cont_cat_correlation_matrix"
    link = "midterm_output/figs/" + name + ".html"
    fig.write_html(link)
    data["Cont/Cat Correlation Matrix"].append(name)


def cat_cat_matrix(df, predictors, cont_predictor_start, data):
    cat_predictors = predictors[:cont_predictor_start]
    matrix = []

    for i in range(len(cat_predictors)):
        cat_predictor_i = cat_predictors[i]
        row = []

        for j in range(len(cat_predictors)):
            cat_predictor_j = cat_predictors[j]
            row.append(cat_correlation(df[cat_predictor_i], df[cat_predictor_j]))

        matrix.append(row)

    fig = px.imshow(
        matrix,
        labels=dict(
            x="Categorical Predictors", y="Categorical Predictors", color="Cramer's V"
        ),
        x=cat_predictors,
        y=cat_predictors,
        title="Categorical vs. Categorical Predictors: Correlation Matrix",
    )

    name = "cat_cat_correlation_matrix"
    link = "midterm_output/figs/" + name + ".html"
    fig.write_html(link)
    data["Cat/Cat Correlation Matrix"].append(name)


def cont_cont_brute_force(df, x_predictor_name, y_predictor_name, response_name):
    x_predictor = df[x_predictor_name]
    y_predictor = df[y_predictor_name]
    response = df[response_name]
    pop_mean = np.mean(response)
    num_bins = 5

    min_value_x = min(x_predictor)
    max_value_x = max(x_predictor)
    full_width_x = abs(max_value_x - min_value_x)
    bin_width_x = full_width_x / num_bins
    lower_bins_x = np.arange(start=min_value_x, stop=max_value_x, step=bin_width_x)
    upper_bins_x = np.arange(
        start=min_value_x + bin_width_x,
        stop=max_value_x + bin_width_x,
        step=bin_width_x,
    )
    bin_centers_x = np.arange(
        start=min_value_x + (bin_width_x / 2), stop=max_value_x, step=bin_width_x
    )
    for i in range(len(bin_centers_x)):
        bin_centers_x[i] = str(bin_centers_x[i])

    min_value_y = min(y_predictor)
    max_value_y = max(y_predictor)
    full_width_y = abs(max_value_y - min_value_y)
    bin_width_y = full_width_y / num_bins
    lower_bins_y = np.arange(start=min_value_y, stop=max_value_y, step=bin_width_y)
    upper_bins_y = np.arange(
        start=min_value_y + bin_width_y,
        stop=max_value_y + bin_width_y,
        step=bin_width_y,
    )
    bin_centers_y = np.arange(
        start=min_value_y + (bin_width_y / 2), stop=max_value_y, step=bin_width_y
    )
    for i in range(len(bin_centers_y)):
        bin_centers_y[i] = str(bin_centers_y[i])

    bin_counts = np.empty((num_bins, num_bins))
    bin_responses = np.empty((num_bins, num_bins))

    for i in range(num_bins):
        for j in range(num_bins):
            bin_counts[i][j] = 0
            bin_responses[i][j] = 0

    for i in range(len(x_predictor)):
        x = x_predictor[i]
        y = y_predictor[i]
        this_response = response[i]

        for bin_x in range(num_bins):
            if x >= lower_bins_x[bin_x] and x <= upper_bins_x[bin_x]:
                break

        for bin_y in range(num_bins):
            if y >= lower_bins_y[bin_y] and y <= upper_bins_y[bin_y]:
                break

        bin_counts[bin_x][bin_y] += 1
        bin_responses[bin_x][bin_y] += this_response

    bin_means = np.empty((num_bins, num_bins))
    bin_residuals = np.empty((num_bins, num_bins))

    for i in range(num_bins):
        for j in range(num_bins):
            if bin_counts[i][j] == 0:
                bin_mean = float("nan")
            else:
                bin_mean = bin_responses[i][j] / bin_counts[i][j]
            bin_means[i][j] = bin_mean
            bin_residuals[i][j] = bin_mean - pop_mean

    bin_means = bin_means.transpose()
    bin_residuals = bin_residuals.transpose()

    fig1 = go.Figure(
        data=go.Heatmap(x=bin_centers_x, y=bin_centers_y, z=bin_means, zmin=0, zmax=1)
    )
    fig1.update_layout(
        title="{} vs. {}: Bin Mean Plot (Pop Mean: {})".format(
            x_predictor_name, y_predictor_name, pop_mean
        ),
        xaxis_title=x_predictor_name,
        yaxis_title=y_predictor_name,
    )

    fig2 = go.Figure(
        data=go.Heatmap(
            x=bin_centers_x, y=bin_centers_y, z=bin_residuals, zmin=-1, zmax=1
        )
    )
    fig2.update_layout(
        title="{} vs. {}: Bin Residual Plot (Pop Mean: {})".format(
            x_predictor_name, y_predictor_name, pop_mean
        ),
        xaxis_title=x_predictor_name,
        yaxis_title=y_predictor_name,
    )
    # fig1.show()
    # fig2.show()


def cont_cat_brute_force(df, cont_predictor_name, cat_predictor_name, response_name):
    cont_predictor = df[cont_predictor_name]
    cat_predictor = df[cat_predictor_name]
    response = df[response_name]
    pop_mean = np.mean(response)
    num_bins_cont = 5
    num_bins_cat = cat_predictor.nunique()

    min_value_cont = min(cont_predictor)
    max_value_cont = max(cont_predictor)
    full_width_cont = abs(max_value_cont - min_value_cont)
    bin_width_cont = full_width_cont / num_bins_cont
    lower_bins_cont = np.arange(
        start=min_value_cont, stop=max_value_cont, step=bin_width_cont
    )
    upper_bins_cont = np.arange(
        start=min_value_cont + bin_width_cont,
        stop=max_value_cont + bin_width_cont,
        step=bin_width_cont,
    )
    bin_centers_cont = np.arange(
        start=min_value_cont + (bin_width_cont / 2),
        stop=max_value_cont,
        step=bin_width_cont,
    )
    for i in range(len(bin_centers_cont)):
        bin_centers_cont[i] = str(bin_centers_cont[i])

    unique_vals = cat_predictor.unique()
    dropped = False

    if not cat_predictor.dtype == "category":
        unique_vals = unique_vals.tolist()

        for i in range(len(unique_vals)):
            if type(unique_vals[i]) is float:
                dropped = unique_vals[i]
                unique_vals.remove(unique_vals[i])
        unique_vals.sort()

        if dropped is not False:
            unique_vals.append(dropped)
            num_bins_cat += 1

    possible_cat_predictors = unique_vals

    bin_counts = np.empty((num_bins_cont, num_bins_cat))
    bin_responses = np.empty((num_bins_cont, num_bins_cat))

    for i in range(num_bins_cont):
        for j in range(num_bins_cat):
            bin_counts[i][j] = 0
            bin_responses[i][j] = 0

    for i in range(len(cont_predictor)):
        x = cont_predictor[i]
        y = cat_predictor[i]
        this_response = response[i]

        for bin_cont in range(num_bins_cont):
            if x >= lower_bins_cont[bin_cont] and x <= upper_bins_cont[bin_cont]:
                break

        for bin_cat in range(num_bins_cat):
            if (
                y is possible_cat_predictors[bin_cat]
                or y == possible_cat_predictors[bin_cat]
            ):
                break

        bin_counts[bin_cont][bin_cat] += 1
        bin_responses[bin_cont][bin_cat] += this_response

    bin_means = np.empty((num_bins_cont, num_bins_cat))
    bin_residuals = np.empty((num_bins_cont, num_bins_cat))

    for i in range(num_bins_cont):
        for j in range(num_bins_cat):
            if bin_counts[i][j] == 0:
                bin_mean = float("nan")
            else:
                bin_mean = bin_responses[i][j] / bin_counts[i][j]
            bin_means[i][j] = bin_mean
            bin_residuals[i][j] = bin_mean - pop_mean

    bin_means = bin_means.transpose()
    bin_residuals = bin_residuals.transpose()

    if dropped is not False:
        possible_cat_predictors.remove(dropped)
        possible_cat_predictors.append("NAN")

    fig1 = go.Figure(
        data=go.Heatmap(
            x=bin_centers_cont, y=possible_cat_predictors, z=bin_means, zmin=0, zmax=1
        )
    )
    fig1.update_layout(
        title="{} vs. {}: Bin Mean Plot (Pop Mean: {})".format(
            cont_predictor_name, cat_predictor_name, pop_mean
        ),
        xaxis_title=cont_predictor_name,
        yaxis_title=cat_predictor_name,
    )

    fig2 = go.Figure(
        data=go.Heatmap(
            x=bin_centers_cont,
            y=possible_cat_predictors,
            z=bin_residuals,
            zmin=-1,
            zmax=1,
        )
    )
    fig2.update_layout(
        title="{} vs. {}: Bin Residual Plot (Pop Mean: {})".format(
            cont_predictor_name, cat_predictor_name, pop_mean
        ),
        xaxis_title=cont_predictor_name,
        yaxis_title=cat_predictor_name,
    )

    # fig1.show()
    fig2.show()


def cat_cat_brute_force(df, x_predictor_name, y_predictor_name, response_name):
    x_predictor = df[x_predictor_name]
    y_predictor = df[y_predictor_name]
    response = df[response_name]
    pop_mean = np.mean(response)
    num_bins_x = x_predictor.nunique()
    num_bins_y = y_predictor.nunique()

    unique_vals_x = x_predictor.unique()
    dropped_x = False

    if not x_predictor.dtype == "category":
        unique_vals_x = unique_vals_x.tolist()

        for i in range(len(unique_vals_x)):
            if type(unique_vals_x[i]) is float:
                dropped_x = unique_vals_x[i]
                unique_vals_x.remove(unique_vals_x[i])
        unique_vals_x.sort()

        if dropped_x is not False:
            unique_vals_x.append(dropped_x)
            num_bins_x += 1

    possible_x_predictors = unique_vals_x

    unique_vals_y = y_predictor.unique()
    dropped_y = False

    if not y_predictor.dtype == "category":
        unique_vals_y = unique_vals_y.tolist()

        for i in range(len(unique_vals_y)):
            if type(unique_vals_y[i]) is float:
                dropped_y = unique_vals_y[i]
                unique_vals_y.remove(unique_vals_y[i])
        unique_vals_y.sort()

        if dropped_y is not False:
            unique_vals_y.append(dropped_y)
            num_bins_y += 1

    possible_y_predictors = unique_vals_y

    bin_counts = np.empty((num_bins_x, num_bins_y))
    bin_responses = np.empty((num_bins_x, num_bins_y))

    for i in range(num_bins_x):
        for j in range(num_bins_y):
            bin_counts[i][j] = 0
            bin_responses[i][j] = 0

    for i in range(len(x_predictor)):
        x = x_predictor[i]
        y = y_predictor[i]
        this_response = response[i]

        for bin_x in range(num_bins_x):
            if x is possible_x_predictors[bin_x] or x == possible_x_predictors[bin_x]:
                break

        for bin_y in range(num_bins_y):
            if y is possible_y_predictors[bin_y] or y == possible_y_predictors[bin_y]:
                break

        bin_counts[bin_x][bin_y] += 1
        bin_responses[bin_x][bin_y] += this_response

    bin_means = np.empty((num_bins_x, num_bins_y))
    bin_residuals = np.empty((num_bins_x, num_bins_y))

    for i in range(num_bins_x):
        for j in range(num_bins_y):
            if bin_counts[i][j] == 0:
                bin_mean = float("nan")
            else:
                bin_mean = bin_responses[i][j] / bin_counts[i][j]
            bin_means[i][j] = bin_mean
            bin_residuals[i][j] = bin_mean - pop_mean

    bin_means = bin_means.transpose()
    bin_residuals = bin_residuals.transpose()

    if dropped_x is not False:
        possible_x_predictors.remove(dropped_x)
        possible_x_predictors.append("NAN")

    if dropped_y is not False:
        possible_y_predictors.remove(dropped_y)
        possible_y_predictors.append("NAN")

    fig1 = go.Figure(
        data=go.Heatmap(
            x=possible_x_predictors,
            y=possible_y_predictors,
            z=bin_means,
            zmin=0,
            zmax=1,
        )
    )
    fig1.update_layout(
        title="{} vs. {}: Bin Mean Plot (Pop Mean: {})".format(
            x_predictor_name, y_predictor_name, pop_mean
        ),
        xaxis_title=x_predictor_name,
        yaxis_title=y_predictor_name,
    )

    fig2 = go.Figure(
        data=go.Heatmap(
            x=possible_x_predictors,
            y=possible_y_predictors,
            z=bin_residuals,
            zmin=-1,
            zmax=1,
        )
    )
    fig2.update_layout(
        title="{} vs. {}: Bin Residual Plot (Pop Mean: {})".format(
            x_predictor_name, y_predictor_name, pop_mean
        ),
        xaxis_title=x_predictor_name,
        yaxis_title=y_predictor_name,
    )

    fig1.show()
    # fig2.show()


def make_clickable(name):
    link = "file:///" + os.getcwd() + "/midterm_output/figs/" + name + ".html"
    return '<a href="{}" target="_blank">{}</a>'.format(link, name)


def main():
    # Step 7 - Put links to original plots done in HW4
    out_dir_exist = os.path.exists("midterm_output/figs")
    if out_dir_exist is False:
        os.makedirs("midterm_output/figs")

    # Step 1 - Given DataFrame, predictors, and response
    test_data_set = get_test_data_set("titanic")
    df = test_data_set[0]
    df = df.reset_index()
    predictors = test_data_set[1]
    response = test_data_set[2]
    df = df.reset_index()

    # Step 2 - Split dataset
    predictors, cont_predictor_start = split_dataset(df, predictors)

    # Step 3+6+9 - Correlation metrics cont/cont pairs + put in table + brute force
    cont_cont_correlation_data = {
        "Predictors (Cont/Cont)": [],
        "Pearson's r": [],
        "Absolute Value of Correlation": [],
        "Scatter Plot": [],
    }
    cont_cont_correlation(
        df, predictors, cont_predictor_start, cont_cont_correlation_data, response
    )
    cont_cont_correlation_df = pd.DataFrame(cont_cont_correlation_data)
    cont_cont_correlation_df = cont_cont_correlation_df.sort_values(
        by=["Absolute Value of Correlation"], ascending=False
    )
    cont_cont_styler = cont_cont_correlation_df.style.format(
        {"Scatter Plot": make_clickable}
    )

    # Step 4+6+10 - Correlation metrics cont/cat pairs + put in table + brute force
    cont_cat_correlation_data = {
        "Predictors (Cont/Cat)": [],
        "Correlation Ratio": [],
        "Absolute Value of Correlation": [],
        "Distribution Plot": [],
        "Violin Plot": [],
    }
    cont_cat_correlation(
        df, predictors, cont_predictor_start, cont_cat_correlation_data, response
    )
    cont_cat_correlation_df = pd.DataFrame(cont_cat_correlation_data)
    cont_cat_correlation_df = cont_cat_correlation_df.sort_values(
        by=["Absolute Value of Correlation"], ascending=False
    )
    cont_cat_styler = cont_cat_correlation_df.style.format(
        {"Distribution Plot": make_clickable, "Violin Plot": make_clickable}
    )

    # Step 5+6+11 - Correlation metrics  cat/cat pairs + put in table + brute force
    cat_cat_correlation_data = {
        "Predictors (Cat/Cat)": [],
        "Cramer's V": [],
        "Absolute Value of Correlation": [],
        "Heatmap": [],
    }
    cat_cat_correlation(
        df, predictors, cont_predictor_start, cat_cat_correlation_data, response
    )
    cat_cat_correlation_df = pd.DataFrame(cat_cat_correlation_data)
    cat_cat_correlation_df = cat_cat_correlation_df.sort_values(
        by=["Absolute Value of Correlation"], ascending=False
    )

    cat_cat_styler = cat_cat_correlation_df.style.format({"Heatmap": make_clickable})

    # Step 8 - Generate correlation matrices
    matrix_data = {
        "Cont/Cont Correlation Matrix": [],
        "Cont/Cat Correlation Matrix": [],
        "Cat/Cat Correlation Matrix": [],
    }
    cont_cont_matrix(df, predictors, cont_predictor_start, matrix_data)
    cont_cat_matrix(df, predictors, cont_predictor_start, matrix_data)
    cat_cat_matrix(df, predictors, cont_predictor_start, matrix_data)

    matrix_df = pd.DataFrame(matrix_data)

    matrix_styler = matrix_df.style.format(
        {
            "Cont/Cont Correlation Matrix": make_clickable,
            "Cont/Cat Correlation Matrix": make_clickable,
            "Cat/Cat Correlation Matrix": make_clickable,
        }
    )

    html = (
        cont_cont_styler.to_html()
        + "\n\n"
        + cont_cat_styler.to_html()
        + "\n\n"
        + cat_cat_styler.to_html()
        + "\n\n"
        + matrix_styler.to_html()
    )

    with open("midterm_output/report.html", "w+") as file:
        file.write(html)
    file.close()

    filename = "file:///" + os.getcwd() + "/midterm_output/report.html"
    webbrowser.open_new_tab(filename)


if __name__ == "__main__":
    sys.exit(main())
