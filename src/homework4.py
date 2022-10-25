import os
import sys
import webbrowser

import numpy as np
import pandas as pd
import seaborn
import statsmodels.api as sm
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

pd.set_option("mode.use_inf_as_na", True)


def response_type_check(df, response):
    # Determine if response is continuous or boolean
    unique_responses = df[response].nunique()

    if unique_responses == 2:
        boolean_check = True
    else:
        boolean_check = False

    return boolean_check


def cat_cont_check(df, predictor):
    # Determine if the predictor is cat/cont

    if df[predictor].dtype == float or (
        (float(df[predictor].nunique()) / len(df[predictor])) > 0.05
    ):
        cat_check = False
    else:
        cat_check = True

    return cat_check


def bool_response_cat_predictor_plots(df, predictor, response):
    df.sort_values(by=[predictor])

    fig = px.density_heatmap(
        df,
        x=predictor,
        y=response,
        title="{} v. {}: Heatmap".format(predictor, response),
    )
    fig.write_html("output/figs/" + predictor + "/heatmap.html")


def bool_response_cont_predictor_plots(df, predictor, response):
    # Distribution Plot
    responses = df[response].unique()
    response1 = responses[0]
    response2 = responses[1]

    response1_predictor = df[df[response] == response1][predictor].dropna()
    response2_predictor = df[df[response] == response2][predictor].dropna()

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
    fig1.write_html("output/figs/" + predictor + "/distribution_plot.html")
    fig2.write_html("output/figs/" + predictor + "/violin_plot.html")


def cont_response_cat_predictor_plots(df, predictor, response):
    # Distribution Plot
    predictors = df[predictor].unique()
    if not df[predictor].dtype == "category":
        predictors.sort()
    hist_data = []
    group_labels = []

    for i in range(len(predictors)):
        predictor_i = predictors[i]
        predictor_i_response = df[df[predictor] == predictor_i][response].dropna()
        hist_data.append(predictor_i_response)
        group_labels.append("Predictor = {}".format(predictor_i))

    fig1 = ff.create_distplot(hist_data, group_labels)
    fig1.update_layout(
        title="{} v. {}: Distribution Plot".format(predictor, response),
        xaxis_title=response,
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
        title="{} vs. {}: Violin Plot".format(predictor, response),
        xaxis_title=predictor,
        yaxis_title=response,
    )
    fig1.write_html("output/figs/" + predictor + "/distribution_plot.html")
    fig2.write_html("output/figs/" + predictor + "/violin_plot.html")


def cont_response_cont_predictor_plots(df, predictor, response):
    fig = px.scatter(df, x=predictor, y=response, trendline="ols")
    fig.update_layout(
        title="{} vs. {}: Scatter Plot", xaxis_title=predictor, yaxis_title=response
    )
    fig.write_html("output/figs/" + predictor + "/scatter_plot.html")


def logistic_regression(df, predictor_name, response):
    x = df[predictor_name]
    y = df[response]

    # Remaking df with just these two columns to remove na's
    temp_df = pd.DataFrame({predictor_name: x, response: y})
    temp_df = temp_df.dropna()
    x = temp_df[predictor_name]

    unique_responses = temp_df[response].unique()
    temp_df.loc[df[response] == unique_responses[0], response] = int(0)
    temp_df.loc[df[response] == unique_responses[1], response] = int(1)
    y = temp_df[response].map(int)

    predictor = sm.add_constant(x)
    logistic_regression_model = sm.Logit(np.asarray(y), np.asarray(predictor))
    logistic_regression_model_fitted = logistic_regression_model.fit()
    t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
    p_value = "{:6e}".format(logistic_regression_model_fitted.pvalues[1])
    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {predictor_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {predictor_name}",
        yaxis_title="y - 0: {}, 1:{}".format(unique_responses[0], unique_responses[1]),
    )
    fig.write_html("output/figs/" + predictor_name + "/logistic_regression.html")
    return t_value, p_value


def linear_regression(df, predictor_name, response):
    x = df[predictor_name]
    y = df[response]

    # Remaking df with just these two columns to remove na's
    temp_df = pd.DataFrame({predictor_name: x, response: y})
    temp_df = temp_df.dropna()
    x = temp_df[predictor_name]
    y = temp_df[response]

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
    fig.write_html("output/figs/" + predictor_name + "/linear_regression.html")
    return t_value, p_value


def diff_with_mean_of_resp(df, predictor_name, response_name, cat_check, bool_check):
    df = pd.DataFrame(
        {predictor_name: df[predictor_name], response_name: df[response_name]}
    )

    if bool_check is True:
        unique_responses = df[response_name].unique()
        df.loc[df[response_name] == unique_responses[0], response_name] = 0
        df.loc[df[response_name] == unique_responses[1], response_name] = 1

    df.sort_values(by=predictor_name)

    predictor = df[predictor_name]
    response = df[response_name]

    bin_means = []
    bin_mean_minus_pop_mean = []
    mean_squared_diff = []
    weighted_mean_squared_diff = []
    dropped = False

    if cat_check is True:
        num_bins = df[predictor_name].nunique()
        unique_vals = df[predictor_name].unique()
        if not df[predictor_name].dtype == "category":
            unique_vals = unique_vals.tolist()

            for i in range(len(unique_vals)):
                if type(unique_vals[i]) is float:
                    dropped = unique_vals[i]
                    unique_vals.remove(unique_vals[i])
            unique_vals.sort()

            if dropped is not False:
                unique_vals.append(dropped)
                num_bins += 1

        possible_cat_predictors = unique_vals
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

    for i in range(len(predictor)):
        cur_val = predictor[i]
        cur_response = response[i]

        for bin_num in range(num_bins):
            if (
                cat_check is True
                and (
                    (cur_val is possible_cat_predictors[bin_num])
                    or cur_val == possible_cat_predictors[bin_num]
                )
            ) or (
                cat_check is False
                and cur_val >= lower_bins[bin_num]
                and cur_val <= upper_bins[bin_num]
            ):
                bin_counts[bin_num] += 1
                bin_responses[bin_num] += cur_response
                break

    population_mean = np.mean(response)

    for i in range(num_bins):
        if bin_counts[i] == 0:
            bin_mean = 0
        else:
            bin_mean = bin_responses[i] / bin_counts[i]
        bin_means.append(bin_mean)
        bin_mean_minus_pop_mean.append(bin_mean - population_mean)
        mean_squared_diff.append((bin_mean - population_mean) ** 2)

        weight = bin_counts[i] / len(predictor)
        weighted_mean_squared_diff.append(
            (weight * ((bin_mean - population_mean) ** 2))
        )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if cat_check is True:
        if dropped is not False:
            possible_cat_predictors.remove(dropped)
            possible_cat_predictors.append("NAN")

        fig.add_trace(
            go.Bar(x=possible_cat_predictors, y=bin_counts, name="Population"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=possible_cat_predictors,
                y=bin_mean_minus_pop_mean,
                name="\u03BC(i) - \u03BC(population)",
            ),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                x=possible_cat_predictors,
                y=np.repeat(population_mean, num_bins),
                name="\u03BC(population)",
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

        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=np.repeat(population_mean, num_bins),
                name="\u03BC(population)",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title_text="Difference with Mean of Response - Predictor {}".format(
            predictor_name
        )
    )

    fig.update_xaxes(title_text="Predictor Bin")
    fig.update_yaxes(title_text="Population", secondary_y=False)
    if bool_check is True:
        fig.update_yaxes(
            title_text="Response- 0: {}, 1: {}".format(
                unique_responses[0], unique_responses[1]
            ),
            secondary_y=True,
            range=[-(max(response)), max(response)],
        )
    else:
        fig.update_yaxes(
            title_text="Response",
            secondary_y=True,
            range=[-(max(response)), max(response)],
        )
    fig.update_layout(
        dict(
            yaxis2={"anchor": "x", "overlaying": "y", "side": "left"},
            yaxis={"anchor": "x", "domain": [0.0, 1.0], "side": "right"},
        )
    )

    dif_w_mean_of_resp = np.mean(mean_squared_diff)

    fig.write_html("output/figs/" + predictor_name + "/diff_w_mean_of_resp.html")
    return dif_w_mean_of_resp, sum(weighted_mean_squared_diff)


def random_forest(df, cont_list, response, bool_check):
    cont_list.append(response)

    cont_df = df[cont_list]
    cont_df = cont_df.dropna()

    X = cont_df[cont_list]
    X = X.drop(response, axis=1)
    y = cont_df[response]
    cont_list = cont_list.remove(response)

    if bool_check is True:
        classifier = RandomForestClassifier()
    else:
        classifier = RandomForestRegressor()

    classifier.fit(X, y)

    feature_imp = pd.Series(
        classifier.feature_importances_, index=cont_list
    ).sort_values(ascending=False)
    return feature_imp


def make_clickable(val):
    if val != "NA":
        return '<a target="_blank" href="{}">{}</a>'.format(val, val)
    else:
        return "NA"


def main():
    # Given a pandas dataframe
    # Contains both a response and predictors
    # Replace with other datasets to test
    df = seaborn.load_dataset(name="titanic")
    predictors = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
    response = "survived"

    out_dir_exist = os.path.exists("output/figs")
    if out_dir_exist is False:
        os.makedirs("output/figs")

    cwd = os.getcwd()

    # Determine if response is continuous or boolean
    bool_check = response_type_check(df, response)

    if bool_check is True:
        response_name = "boolean"
    else:
        response_name = "continuous"

    response_col = []
    predictor_col = []
    heatmap_col = []
    distribution_plot_col = []
    violin_plot_col = []
    scatter_plot_col = []
    t_score_col = []
    p_value_col = []
    logistic_regression_col = []
    linear_regression_col = []
    rf_var_imp_col = []
    mwr_unweighted_col = []
    mwr_weighted_col = []
    mwr_plot_col = []

    cont_list = []  # will store continuous predictors for random forest
    counter = -1
    counter_list = []

    # Loop through each predictor column
    for predictor in predictors:
        counter += 1

        out_dir_exist = os.path.exists("output/figs/{}".format(predictor))
        if out_dir_exist is False:
            os.makedirs("output/figs/{}".format(predictor))

        response_col.append(response)

        # Determine if the predictor is cat/cont
        cat_check = cat_cont_check(df, predictor)

        if cat_check is False:
            cont_list.append(predictor)
            counter_list.append(counter)
            predictor_type_string = "cont"
        else:
            predictor_type_string = "cat"

        predictor_col.append(predictor + "({})".format(predictor_type_string))

        # Automatically generate the necessary plot(s) to inspect it
        # Calculate the different ranking algos
        # p-value & t-score (continuous predictors only) along with it's plot
        # Regression: Continuous response
        # Logistic regression: Boolean response
        if bool_check is True and cat_check is True:
            bool_response_cat_predictor_plots(df, predictor, response)
            heatmap_col.append(
                "file:///" + cwd + "/output/figs/" + predictor + "/heatmap.html"
            )
            distribution_plot_col.append("NA")
            violin_plot_col.append("NA")
            scatter_plot_col.append("NA")

            t_score_col.append("NA")
            p_value_col.append("NA")
            logistic_regression_col.append("NA")
            linear_regression_col.append("NA")
        elif bool_check is True and cat_check is False:
            bool_response_cont_predictor_plots(df, predictor, response)
            heatmap_col.append("NA")
            distribution_plot_col.append(
                "file:///"
                + cwd
                + "/output/figs/"
                + predictor
                + "/distribution_plot.html"
            )
            violin_plot_col.append(
                "file:///" + cwd + "/output/figs/" + predictor + "/violin_plot.html"
            )
            scatter_plot_col.append("NA")

            t_score, p_value = logistic_regression(df, predictor, response)
            t_score_col.append(t_score)
            p_value_col.append(p_value)
            logistic_regression_col.append(
                "file:///"
                + cwd
                + "/output/figs/"
                + predictor
                + "/logistic_regression.html"
            )
            linear_regression_col.append("NA")
        elif bool_check is False and cat_check is True:
            cont_response_cat_predictor_plots(df, predictor, response)
            heatmap_col.append("NA")
            distribution_plot_col.append(
                "file:///"
                + cwd
                + "/output/figs/"
                + predictor
                + "/distribution_plot.html"
            )
            violin_plot_col.append(
                "file:///" + cwd + "/output/figs/" + predictor + "/violin_plot.html"
            )
            scatter_plot_col.append("NA")

            t_score_col.append("NA")
            p_value_col.append("NA")
            logistic_regression_col.append("NA")
            linear_regression_col.append("NA")
        else:
            cont_response_cont_predictor_plots(df, predictor, response)
            heatmap_col.append("NA")
            distribution_plot_col.append("NA")
            violin_plot_col.append("NA")
            scatter_plot_col.append(
                "file:///" + cwd + "/output/figs/" + predictor + "/scatter_plot.html"
            )

            t_score, p_value = linear_regression(df, predictor, response)
            t_score_col.append(t_score)
            p_value_col.append(p_value)
            logistic_regression_col.append("NA")
            linear_regression_col.append(
                "file:///"
                + cwd
                + "/output/figs/"
                + predictor
                + "/linear_regression.html"
            )

        """
        # Calculate the different ranking algos
        # p-value & t-score (continuous predictors only) along with it's plot
        # Regression: Continuous response
        # Logistic regression: Boolean response
        if cat_check is False:
            if bool_check is True:
                logistic_regression(df, predictor, response)
            else:
                linear_regression(df, predictor, response)
        """

        # Difference with mean of response along with it's plot (weighted and unweighted)
        unweighted, weighted = diff_with_mean_of_resp(
            df, predictor, response, cat_check, bool_check
        )
        mwr_unweighted_col.append(unweighted)
        mwr_weighted_col.append(weighted)
        mwr_plot_col.append(
            "file:///" + cwd + "/output/figs/" + predictor + "/diff_w_mean_of_resp.html"
        )

        rf_var_imp_col.append("NA")

    # Random Forest Variable importance ranking (continuous predictors only)
    importance = random_forest(df, cont_list, response, bool_check)

    counter = -1

    for i in counter_list:
        counter += 1
        rf_var_imp_col[i] = importance[counter]

    data = {
        "Response ({})".format(response_name): response_col,
        "Predictor": predictor_col,
        "Heatmap": heatmap_col,
        "Distribution Plot": distribution_plot_col,
        "Violin Plot": violin_plot_col,
        "Scatter Plot": scatter_plot_col,
        "t-score": t_score_col,
        "p-value": p_value_col,
        "Logistic Regression Plot": logistic_regression_col,
        "Linear Regression Plot": linear_regression_col,
        "RF VarImp": rf_var_imp_col,
        "MWR Unweighted": mwr_unweighted_col,
        "MWR Weighted": mwr_weighted_col,
        "MWR Plot": mwr_plot_col,
    }
    df = pd.DataFrame(data)

    styler = df.style.format(
        {
            "Heatmap": make_clickable,
            "Distribution Plot": make_clickable,
            "Violin Plot": make_clickable,
            "Scatter Plot": make_clickable,
            "Logistic Regression Plot": make_clickable,
            "Linear Regression Plot": make_clickable,
            "MWR Plot": make_clickable,
        }
    )

    html = styler.to_html()

    with open("output/report.html", "w+") as file:
        file.write(html)
    file.close()

    filename = "file:///" + os.getcwd() + "/" + "output/report.html"
    webbrowser.open_new_tab(filename)


if __name__ == "__main__":
    sys.exit(main())
