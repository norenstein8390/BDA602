import math

import numpy as np
from plotly import graph_objects as go


class MidtermBruteForce:
    def __init__(self, df, response):
        self.df = df
        self.response = response

    def cont_cont_diff_of_mean(self, x_predictor_name, y_predictor_name):
        x_predictor = self.df[x_predictor_name]
        y_predictor = self.df[y_predictor_name]
        response = self.df[self.response]
        pop_mean = np.mean(response)
        num_bins = 10

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
        bin_means_text = np.empty((num_bins, num_bins))
        bin_means_text = bin_means_text.astype(str)
        bin_residuals_text = np.empty((num_bins, num_bins))
        bin_residuals_text = bin_residuals_text.astype(str)

        for i in range(num_bins):
            for j in range(num_bins):
                if bin_counts[i][j] == 0:
                    bin_mean = float("nan")
                    residual = float("nan")
                else:
                    bin_mean = bin_responses[i][j] / bin_counts[i][j]
                    residual = bin_mean - pop_mean
                bin_means[i][j] = bin_mean
                bin_residuals[i][j] = residual
                bin_means_text[i][j] = "{} (Count: {})".format(
                    str(round(bin_mean, 3)), str(int(bin_counts[i][j]))
                )
                bin_residuals_text[i][j] = "{} (Count: {})".format(
                    str(round(residual, 3)), str(int(bin_counts[i][j]))
                )

        bin_means = bin_means.transpose()
        bin_residuals = bin_residuals.transpose()
        bin_means_text = bin_means_text.transpose()
        bin_residuals_text = bin_residuals_text.transpose()

        fig1 = go.Figure(
            data=go.Heatmap(
                x=bin_centers_x,
                y=bin_centers_y,
                z=bin_means,
                zmin=np.min(bin_means),
                zmax=np.max(bin_means),
            )
        )
        fig1.update_layout(
            title=f"{x_predictor_name} vs. {y_predictor_name}: Bin Mean Plot (Pop Mean: {pop_mean})",
            xaxis_title=x_predictor_name,
            yaxis_title=y_predictor_name,
        )
        fig1.update_traces(text=bin_means_text, texttemplate="%{text}")

        fig2 = go.Figure(
            data=go.Heatmap(
                x=bin_centers_x,
                y=bin_centers_y,
                z=bin_residuals,
                zmin=np.min(bin_residuals),
                zmax=np.max(bin_residuals),
            )
        )
        fig2.update_layout(
            title=f"{x_predictor_name} vs. {y_predictor_name}: Bin Residual Plot (Pop Mean: {pop_mean})",
            xaxis_title=x_predictor_name,
            yaxis_title=y_predictor_name,
        )
        fig2.update_traces(text=bin_residuals_text, texttemplate="%{text}")

        name_bin_plot = f"{x_predictor_name}_{y_predictor_name}_bin_mean_plot"
        link_bin_plot = f"final/midterm_output/figs/{name_bin_plot}.html"
        fig1.write_html(link_bin_plot)

        name_residual_plot = f"{x_predictor_name}_{y_predictor_name}_bin_residual_plot"
        link_residual_plot = f"final/midterm_output/figs/{name_residual_plot}.html"
        fig2.write_html(link_residual_plot)

        bin_residuals = bin_residuals.transpose()

        for i in range(num_bins):
            for j in range(num_bins):
                if math.isnan(bin_residuals[i][j]):
                    bin_residuals[i][j] = 0
                else:
                    bin_residuals[i][j] = bin_residuals[i][j] ** 2

        bin_weighted = np.empty((num_bins, num_bins))

        for i in range(num_bins):
            for j in range(num_bins):
                weight = bin_counts[i][j] / len(x_predictor)
                bin_weighted[i][j] = bin_residuals[i][j] * weight

        unweighted = (1 / (num_bins * num_bins)) * np.sum(bin_residuals)
        weighted = np.sum(bin_weighted)
        return unweighted, weighted, name_bin_plot, name_residual_plot

    def cont_cont_brute_force(self, predictors, cont_predictor_start, data):
        cont_predictors = predictors[cont_predictor_start:]

        for i in range(len(cont_predictors)):
            x_predictor = cont_predictors[i]

            for j in range(i + 1, len(cont_predictors)):
                y_predictor = cont_predictors[j]
                data["Predictors (Cont/Cont)"].append(
                    x_predictor + " and " + y_predictor
                )
                (
                    unweighted,
                    weighted,
                    bin_plot,
                    residual_plot,
                ) = self.cont_cont_diff_of_mean(x_predictor, y_predictor)
                data["Difference of Mean Response"].append(unweighted)
                data["Weighted Difference of Mean Response"].append(weighted)
                data["Bin Mean Plot"].append(bin_plot)
                data["Bin Residual Plot"].append(residual_plot)
