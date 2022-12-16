import numpy as np
import pandas as pd
import statsmodels.api as sm
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier


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

        y = temp_df[self.response].map(int)

        predictor = sm.add_constant(x)
        logistic_regression_model = sm.Logit(np.asarray(y), np.asarray(predictor))
        logistic_regression_model_fitted = logistic_regression_model.fit()
        t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
        p_value = round(logistic_regression_model_fitted.pvalues[1], 6)
        fig = px.scatter(x=x, y=y, trendline="ols")
        fig.update_layout(
            title=f"Variable: {predictor_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {predictor_name}",
            yaxis_title="Response",
        )
        fig.write_html(
            f"final/hw4_output/figs/{predictor_name}_logistic_regression.html"
        )
        return t_value, p_value

    def diff_with_mean_of_resp(self, predictor_name, cat_check):
        df = pd.DataFrame(
            {
                predictor_name: self.df[predictor_name],
                self.response: self.df[self.response],
            }
        )

        df = df.sort_values(by=predictor_name)
        df = df.reset_index()

        predictor = df[predictor_name]
        response = df[self.response]

        bin_means = []
        bin_mean_minus_pop_mean = []
        mean_squared_diff = []
        weighted_mean_squared_diff = []

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
                    cat_check is False
                    and bin_num > 0
                    and cur_val >= lower_bins[bin_num]
                    and cur_val <= upper_bins[bin_num]
                ) or (
                    cat_check is False
                    and cur_val >= lower_bins[bin_num]
                    and cur_val < upper_bins[bin_num]
                ):
                    bin_counts[bin_num] += 1
                    bin_responses[bin_num] += cur_response
                    break

        if (
            bin_counts[0] / len(predictor) < 0.05
            or bin_counts[9] / len(predictor) < 0.05
        ):
            print(predictor)
            print(predictor[0])
            first_bin_end = predictor[int(len(predictor) * 0.05)]
            last_bin_start = predictor[int(len(predictor) * 0.95)]
            full_width = abs(last_bin_start - first_bin_end)
            bin_width = full_width / 8
            lower_bins = [min_value, first_bin_end]
            upper_bins = [first_bin_end]
            bin_centers = [
                (min_value + first_bin_end) / 2,
                first_bin_end + bin_width / 2,
            ]
            widths = [
                abs(first_bin_end - min_value),
                bin_width,
                bin_width,
                bin_width,
                bin_width,
                bin_width,
                bin_width,
                bin_width,
                bin_width,
                abs(max_value - last_bin_start),
            ]

            for i in range(7):
                lower_bins.append(lower_bins[i + 1] + bin_width)
                upper_bins.append(upper_bins[i] + bin_width)
                bin_centers.append(bin_centers[i + 1] + bin_width)

            lower_bins.append(last_bin_start)
            upper_bins.append(last_bin_start)
            upper_bins.append(max_value)
            bin_centers.append(last_bin_start + abs(max_value - last_bin_start) / 2)

            bin_counts = np.repeat(0, num_bins)
            bin_responses = np.repeat(0, num_bins)
            print(lower_bins)
            print(upper_bins)
            print(bin_centers)

            for i in range(len(predictor)):
                cur_val = predictor[i]
                cur_response = response[i]

                for bin_num in range(num_bins):
                    if (
                        cat_check is False
                        and bin_num > 0
                        and cur_val >= lower_bins[bin_num]
                        and cur_val <= upper_bins[bin_num]
                    ) or (
                        cat_check is False
                        and cur_val >= lower_bins[bin_num]
                        and cur_val < upper_bins[bin_num]
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

        fig.add_trace(
            go.Bar(x=bin_centers, y=bin_counts, width=widths, name="Population"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=bin_means,
                name="\u03BC(i)",
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

        fig.update_yaxes(
            title_text="Response",
            secondary_y=True,
            range=[min(response), max(response)],
        )

        fig.update_layout(
            dict(
                yaxis2={"anchor": "x", "overlaying": "y", "side": "left"},
                yaxis={"anchor": "x", "domain": [0.0, 1.0], "side": "right"},
            )
        )

        dif_w_mean_of_resp = np.mean(mean_squared_diff)
        fig.write_html(
            f"final/hw4_output/figs/{predictor_name}_diff_w_mean_of_resp.html"
        )
        return dif_w_mean_of_resp, sum(weighted_mean_squared_diff)

    def random_forest(self, cont_list):
        cont_list.append(self.response)

        cont_df = self.df[cont_list]
        cont_df = cont_df.dropna()

        X = cont_df[cont_list]
        X = X.drop(self.response, axis=1)
        y = cont_df[self.response]
        cont_list = cont_list.remove(self.response)

        classifier = RandomForestClassifier()

        classifier.fit(X, y)

        feature_imp = pd.Series(
            classifier.feature_importances_, index=cont_list
        ).sort_values(ascending=False)
        return feature_imp
