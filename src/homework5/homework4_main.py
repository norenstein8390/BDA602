import os

import pandas as pd
from homework4_plots import Homework4Plotter
from homework4_scores import Homework4Scorer


class Homework4ReportMaker:
    def __init__(self, df, predictors, response):
        self.df = df
        self.predictors = predictors
        self.response = response
        self.response_name = "boolean"
        self.plotter = Homework4Plotter(df, response)
        self.scorer = Homework4Scorer(df, response)

        out_dir_exist = os.path.exists("final/hw4_output/figs")

        if out_dir_exist is False:
            os.makedirs("final/hw4_output/figs")

    def make_clickable(self, name):
        link = f"file:///{os.getcwd()}/final/hw4_output/figs/{name}.html"
        return f'<a href="{link}" target="_blank">{name}</a>'

    def make_html(self, dict):
        df = pd.DataFrame(dict)
        df = df.sort_values(by=["t-score abs"], ascending=False)

        styler = df.style.format(
            {
                "Heatmap": self.make_clickable,
                "Distribution Plot": self.make_clickable,
                "Violin Plot": self.make_clickable,
                "Scatter Plot": self.make_clickable,
                "Logistic Regression Plot": self.make_clickable,
                "Linear Regression Plot": self.make_clickable,
                "MWR Plot": self.make_clickable,
            }
        )

        html = "<h1>Homework 5 Report</h1>\n\n" + "<h2>Plots and Rankings</h2>\n\n"
        html += styler.to_html()
        html += "\n\n"
        return html

    def make_plots_rankings(self):
        response_col = []
        predictor_col = []
        distribution_plot_col = []
        violin_plot_col = []
        t_score_col = []
        p_value_col = []
        logistic_regression_col = []
        rf_var_imp_col = []
        mwr_unweighted_col = []
        mwr_weighted_col = []
        mwr_plot_col = []

        cont_list = []  # will store continuous predictors for random forest
        counter = -1
        counter_list = []

        # Loop through each predictor column
        for predictor in self.predictors:
            counter += 1

            out_dir_exist = os.path.exists(f"final/hw4_output/figs/{predictor}")
            if out_dir_exist is False:
                os.makedirs(f"final/hw4_output/figs/{predictor}")

            response_col.append(self.response)

            # Determine if the predictor is cat/cont
            cat_check = False

            cont_list.append(predictor)
            counter_list.append(counter)
            predictor_type_string = "cont"

            predictor_col.append(predictor + f"({predictor_type_string})")

            # Automatically generate the necessary plot(s) to inspect it
            # Calculate the different ranking algos
            # p-value & t-score (continuous predictors only) along with it's plot
            # Regression: Continuous response
            # Logistic regression: Boolean response
            self.plotter.bool_response_cont_predictor_plots(predictor)
            distribution_plot_col.append(f"{predictor}_distribution_plot")
            violin_plot_col.append(f"{predictor}_violin_plot")
            t_score, p_value = self.scorer.logistic_regression(predictor)
            t_score_col.append(t_score)
            p_value_col.append(p_value)
            logistic_regression_col.append(f"{predictor}_logistic_regression")

            # Difference with mean of response along with it's plot (weighted and unweighted)
            unweighted, weighted = self.scorer.diff_with_mean_of_resp(
                predictor, cat_check
            )
            mwr_unweighted_col.append(unweighted)
            mwr_weighted_col.append(weighted)
            mwr_plot_col.append(f"{predictor}_diff_w_mean_of_resp")
            rf_var_imp_col.append("NA")

        # Random Forest Variable importance ranking (continuous predictors only)
        importance = self.scorer.random_forest(cont_list)

        counter = -1

        for i in counter_list:
            counter += 1
            rf_var_imp_col[i] = importance[counter]

        data = {
            "Response ({})".format(self.response_name): response_col,
            "Predictor": predictor_col,
            "Distribution Plot": distribution_plot_col,
            "Violin Plot": violin_plot_col,
            "t-score": t_score_col,
            "t-score abs": [abs(score) for score in t_score_col],
            "p-value": p_value_col,
            "Logistic Regression Plot": logistic_regression_col,
            "RF VarImp": rf_var_imp_col,
            "MWR Unweighted": mwr_unweighted_col,
            "MWR Weighted": mwr_weighted_col,
            "MWR Plot": mwr_plot_col,
        }

        return self.make_html(data)
