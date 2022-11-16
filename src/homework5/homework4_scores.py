import numpy as np
import pandas as pd
import statsmodels.api as sm
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class Homework4Scorer:
    def __init__(self, df, response):
        self.df = df
        self.response = response

    def logistic_regression(self, predictor_name):
        x = self.df[predictor_name]
        y = self.df[self.response]

        # Remaking df with just these two columns to remove na's
        temp_df = pd.DataFrame({predictor_name: x, self.response: y})
        temp_df = temp_df.dropna()
        x = temp_df[predictor_name]

        unique_responses = temp_df[self.response].unique()
        temp_df.loc[self.df[self.response] == unique_responses[0], self.response] = int(
            0
        )
        temp_df.loc[self.df[self.response] == unique_responses[1], self.response] = int(
            1
        )
        y = temp_df[self.response].map(int)

        predictor = sm.add_constant(x)
        logistic_regression_model = sm.Logit(np.asarray(y), np.asarray(predictor))
        logistic_regression_model_fitted = logistic_regression_model.fit()
        t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
        p_value = "{:6e}".format(logistic_regression_model_fitted.pvalues[1])
        fig = px.scatter(x=x, y=y, trendline="ols")
        fig.update_layout(
            title=f"Variable: {predictor_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {predictor_name}",
            yaxis_title=f"y - 0: {unique_responses[0]}, 1:{unique_responses[1]}",
        )
        fig.write_html(f"hw4_output/figs/{predictor_name}_logistic_regression.html")
        return t_value, p_value

    def linear_regression(self, predictor_name):
        x = self.df[predictor_name]
        y = self.df[self.response]

        # Remaking df with just these two columns to remove na's
        temp_df = pd.DataFrame({predictor_name: x, self.response: y})
        temp_df = temp_df.dropna()
        x = temp_df[predictor_name]
        y = temp_df[self.response]

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
        fig.write_html(f"hw4_output/figs/{predictor_name}_linear_regression.html")
        return t_value, p_value

    def diff_with_mean_of_resp(self, predictor_name, cat_check, bool_check):
        df = pd.DataFrame(
            {
                predictor_name: self.df[predictor_name],
                self.response: self.df[self.response],
            }
        )

        if bool_check is True:
            unique_responses = df[self.response].unique()
            df.loc[df[self.response] == unique_responses[0], self.response] = 0
            df.loc[df[self.response] == unique_responses[1], self.response] = 1

        df.sort_values(by=predictor_name)

        predictor = df[predictor_name]
        response = df[self.response]

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
                    (
                        cat_check is True
                        and (
                            (cur_val is possible_cat_predictors[bin_num])
                            or cur_val == possible_cat_predictors[bin_num]
                        )
                    )
                    or (
                        cat_check is False
                        and bin_num > 0
                        and cur_val >= lower_bins[bin_num]
                        and cur_val <= upper_bins[bin_num]
                    )
                    or (
                        cat_check is False
                        and cur_val >= lower_bins[bin_num]
                        and cur_val < upper_bins[bin_num]
                    )
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
                go.Bar(x=bin_centers, y=bin_counts, name="Population"),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=bin_means,
                    # y=bin_mean_minus_pop_mean,
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
            title_text=f"Difference with Mean of Response - Predictor {predictor_name}"
        )

        fig.update_xaxes(title_text="Predictor Bin")
        fig.update_yaxes(title_text="Population", secondary_y=False)
        if bool_check is True:
            fig.update_yaxes(
                title_text=f"Response- 0: {unique_responses[0]}, 1: {unique_responses[1]}",
                secondary_y=True,
                range=[min(response), max(response)],
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
        fig.write_html(f"hw4_output/figs/{predictor_name}_diff_w_mean_of_resp.html")
        return dif_w_mean_of_resp, sum(weighted_mean_squared_diff)

    def random_forest(self, cont_list, bool_check):
        cont_list.append(self.response)

        cont_df = self.df[cont_list]
        cont_df = cont_df.dropna()

        X = cont_df[cont_list]
        X = X.drop(self.response, axis=1)
        y = cont_df[self.response]
        cont_list = cont_list.remove(self.response)

        if bool_check is True:
            classifier = RandomForestClassifier()
        else:
            classifier = RandomForestRegressor()

        classifier.fit(X, y)

        feature_imp = pd.Series(
            classifier.feature_importances_, index=cont_list
        ).sort_values(ascending=False)
        return feature_imp
