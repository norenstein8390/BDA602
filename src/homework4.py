import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier


def response_type_check(df, response):
    # Determine if response is continuous or boolean
    unique_responses = df[response].nunique()

    if unique_responses == 2:
        boolean_check = True
        print("Response is boolean\n")
    else:
        boolean_check = False
        print("Response is continuous\n")

    return boolean_check


def cat_cont_check(df, predictor):
    # Determine if the predictor is cat/cont
    if isinstance(df[predictor][0], float):
        cat_check = False
        print("Predictor {} is continuous\n".format(predictor))
    else:
        cat_check = True
        print("Predictor {} is categorical\n".format(predictor))

    return cat_check


def bool_response_cat_predictor_plots(df, predictor, response):
    x = df[predictor]
    x_type = type(x[0])

    # density_heatmap needs same datatypes and this should work
    df[response] = df[response].map(x_type)

    plot = px.density_heatmap(
        df,
        x=predictor,
        y=response,
        title="{} v. {}: Heatmap".format(predictor, response),
    )
    plot.show()


def bool_response_cont_predictor_plots(df, predictor, response):
    # Distribution Plot
    responses = df[response].unique()
    response1 = responses[0]
    response2 = responses[1]

    response1_predictor = df[df[response] == response1][predictor]
    response2_predictor = df[df[response] == response2][predictor]

    pd.set_option("mode.use_inf_as_na", True)
    response1_predictor = response1_predictor.dropna()
    response2_predictor = response2_predictor.dropna()

    hist_data = [response1_predictor, response2_predictor]
    group_labels = [
        "Response = {}".format(response1),
        "Response = {}".format(response2),
    ]

    fig1 = ff.create_distplot(hist_data, group_labels)
    fig1.update_layout(
        title="{} v. {}: Distribution Plot".format(predictor, response),
        xaxis_title=predictor,
        yaxis_title="Distribution",
    )
    fig1.show()

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
        title="{} vs. {}: Violin Plot".format(predictor, response),
        xaxis_title=response,
        yaxis_title=predictor,
    )
    fig2.show()


def cont_response_cat_predictor_plots(df, predictor, response):
    # Distribution Plot
    predictors = df[predictor].unique()
    hist_data = []
    group_labels = []

    for i in range(len(predictors)):
        predictor_i = predictors[i]
        predictor_i_response = df[df[predictor] == predictor_i][response]
        pd.set_option("mode.use_inf_as_na", True)
        predictor_i_response = predictor_i_response.dropna()
        hist_data.append(predictor_i_response)
        group_labels.append("Predictor = {}".format(predictor_i))

    fig1 = ff.create_distplot(hist_data, group_labels)
    fig1.update_layout(
        title="{} v. {}: Distribution Plot".format(predictor, response),
        xaxis_title=response,
        yaxis_title="Distribution",
    )
    fig1.show()

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
        title="{} vs. {}: Violin Plot".format(predictor, response),
        xaxis_title=predictor,
        yaxis_title=response,
    )
    fig2.show()


def cont_response_cont_predictor_plots(df, predictor, response):
    x = df[predictor]
    y = df[response]

    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title="{} vs. {}: Scatter Plot", xaxis_title=predictor, yaxis_title=response
    )
    fig.show()


def logistic_regression(df, predictor_name, response):
    x = df[predictor_name]
    y = df[response]

    # Remaking df with just these two columns to remove na's
    df = pd.DataFrame({predictor_name: x, response: y})
    pd.set_option("mode.use_inf_as_na", True)
    df = df.dropna()
    x = df[predictor_name]
    y = df[response].map(int)

    predictor = sm.add_constant(x)
    logistic_regression_model = sm.Logit(np.asarray(y), np.asarray(predictor))
    logistic_regression_model_fitted = logistic_regression_model.fit()
    t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
    p_value = "{:6e}".format(logistic_regression_model_fitted.pvalues[1])
    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {predictor_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {predictor_name}",
        yaxis_title="y",
    )
    fig.show()


def linear_regression(df, predictor_name, response):
    x = df[predictor_name]
    y = df[response]

    # Remaking df with just these two columns to remove na's
    df = pd.DataFrame({predictor_name: x, response: y})
    pd.set_option("mode.use_inf_as_na", True)
    df = df.dropna()
    x = df[predictor_name]
    y = df[response].map(int)

    predictor = sm.add_constant(x)
    linear_regression_model = sm.OLS(y, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:6e}".format(linear_regression_model_fitted.pvalues[1])
    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {predictor_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {predictor_name}",
        yaxis_title="y",
    )
    fig.show()


def diff_with_mean_of_resp(df, predictor_name, response_name, cat_check, bool_check):
    if bool_check is True:
        unique_responses = df[response_name].unique()
        df.loc[df[response_name] == unique_responses[0], response_name] = 0
        df.loc[df[response_name] == unique_responses[1], response_name] = 1

    predictor = df[predictor_name]
    response = df[response_name]

    bin_means = []
    bin_mean_minus_pop_mean = []
    mean_squared_diff = []
    weighted_mean_squared_diff = []

    if cat_check is True:
        num_bins = df[predictor_name].nunique()
        possible_cat_predictors = df[predictor_name].unique()
    else:
        num_bins = 10

    bin_counts = np.repeat(0, num_bins)
    bin_responses = np.repeat(0, num_bins)

    if cat_check is False:
        min_value = min(predictor)
        max_value = max(predictor)
        full_width = abs(max_value - min_value)
        bin_width = full_width / num_bins
        lower_bins = np.arange(start=min_value, stop=max_value, step=bin_width)
        upper_bins = np.arange(
            start=min_value + bin_width, stop=max_value + bin_width, step=bin_width
        )
        bin_centers = np.arange(
            start=min_value + (bin_width / 2), stop=max_value, step=bin_width
        )
        bin_counts = np.repeat(0, num_bins)
        bin_responses = np.repeat(0, num_bins)

    for i in range(len(predictor)):
        cur_val = predictor[i]
        cur_response = response[i]

        for bin_num in range(num_bins):
            if (cat_check is True and cur_val == possible_cat_predictors[bin_num]) or (
                cat_check is False
                and cur_val >= lower_bins[bin_num]
                and cur_val <= upper_bins[bin_num]
            ):
                bin_counts[bin_num] += 1
                bin_responses[bin_num] += cur_response
                break

    population_mean = np.mean(response)

    for i in range(num_bins):
        bin_mean = bin_responses[i] / bin_counts[i]
        bin_means.append(bin_mean)
        bin_mean_minus_pop_mean.append(bin_mean - population_mean)
        mean_squared_diff.append((bin_mean - population_mean) ** 2)

        weight = bin_counts[i] / len(predictor)
        weighted_mean_squared_diff.append(
            (weight * ((bin_mean - population_mean) ** 2))
        )

    bin_mean_minus_pop_mean = np.array(bin_mean_minus_pop_mean)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if cat_check is True:
        fig.add_trace(
            go.Bar(x=possible_cat_predictors, y=bin_counts, name="Population"),
            secondary_y=False,
        )

        possible_cat_predictors.sort()

        fig.add_trace(
            go.Scatter(
                x=possible_cat_predictors,
                y=bin_mean_minus_pop_mean,
                name="\u03BC(i) - \u03BC(population)",
            ),
            secondary_y=True,
        )
    else:
        fig.add_trace(
            go.Bar(x=bin_centers, y=bin_counts, name="Population"), secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=bin_mean_minus_pop_mean,
                name="\u03BC(i) - \u03BC(population)",
            ),
            secondary_y=True,
        )

    fig.add_hline(y=population_mean, name="population mean")

    fig.update_layout(
        title_text="Difference with Mean of Response - Predictor {}".format(
            predictor_name
        )
    )

    fig.update_xaxes(title_text="Predictor Bin")
    fig.update_yaxes(title_text="Response", secondary_y=False)
    fig.update_yaxes(
        title_text="Population",
        secondary_y=True,
        range=[-(max(response)), max(response)],
    )
    fig.update_layout(
        dict(
            yaxis2={"anchor": "x", "overlaying": "y", "side": "left"},
            yaxis={"anchor": "x", "domain": [0.0, 1.0], "side": "right"},
        )
    )
    fig.show()

    dif_w_mean_of_resp = np.mean(mean_squared_diff)
    print(
        "Difference with mean of response (unweighted) = {}".format(
            str(dif_w_mean_of_resp)
        )
    )
    print(
        "Difference with mean of response (weighted) = {}".format(
            str(sum(weighted_mean_squared_diff))
        )
    )


def random_forest(df, cont_list, response):
    cont_list.append(response)
    pd.set_option("mode.use_inf_as_na", True)

    full_list = df[cont_list]
    full_list = full_list.dropna()

    X = full_list[cont_list]
    X = X.drop(response, axis=1)
    y = full_list[response]
    cont_list = cont_list.remove(response)

    classifier = RandomForestClassifier()
    classifier.fit(X, y)

    feature_imp = pd.Series(
        classifier.feature_importances_, index=cont_list
    ).sort_values(ascending=False)
    print(feature_imp)


def main():
    # Given a pandas dataframe
    # Contains both a response and predictors
    df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    )

    # Given a list of predictors and the response columns
    predictors = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
    ]  # NOTE TO SELF - Test with more predictors in the future (and different type of response)
    response = "survived"

    # Determine if response is continuous or boolean
    bool_check = response_type_check(df, response)

    cont_list = []  # will store continuous predictors for random forest

    # Loop through each predictor column
    for predictor in predictors:
        # Determine if the predictor is cat/cont
        cat_check = cat_cont_check(df, predictor)

        if cat_check is False:
            cont_list.append(predictor)

        # Automatically generate the necessary plot(s) to inspect it
        if bool_check is True and cat_check is True:
            bool_response_cat_predictor_plots(df, predictor, response)
        elif bool_check is True and cat_check is False:
            bool_response_cont_predictor_plots(df, predictor, response)
        elif bool_check is False and cat_check is True:
            cont_response_cat_predictor_plots(df, predictor, response)
        else:
            cont_response_cont_predictor_plots(df, predictor, response)

        # Calculate the different ranking algos
        # p-value & t-scole (continuous predictors only) along with it's plot
        # Regression: Continuous response
        # Logistic regression: Boolean response
        if cat_check is False:
            if bool_check is True:
                logistic_regression(df, predictor, response)
            else:
                linear_regression(df, predictor, response)

        # Difference with mean of response along with it's plot (weighted and unweighted)
        diff_with_mean_of_resp(df, predictor, response, cat_check, bool_check)

    # Random Forest Variable importance ranking (continuous predictors only)
    random_forest(df, cont_list, response)


if __name__ == "__main__":
    sys.exit(main())
