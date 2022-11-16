import os
import webbrowser

import pandas as pd
from homework4_plots import Homework4Plotter
from homework4_scores import Homework4Scorer


class Homework4ReportMaker:
    def __init__(self, df, predictors, response):
        self.df = df
        self.predictors = predictors
        self.response = response
        self.response_name = None
        self.boolean_check = None
        self.plotter = Homework4Plotter(df, response)
        self.scorer = Homework4Scorer(df, response)

        out_dir_exist = os.path.exists("hw4_output/figs")

        if out_dir_exist is False:
            os.makedirs("hw4_output/figs")

    def response_type_check(self, df, response):
        # Determine if response is continuous or boolean
        unique_responses = df[response].nunique()

        if unique_responses == 2:
            self.boolean_check = True
            self.response_name = "boolean"
        else:
            self.boolean_check = False
            self.response_name = "continuous"

    def cat_cont_check(self, df, predictor):
        # Determine if the predictor is cat/cont
        if df[predictor].dtype == float or df[predictor].dtype == int:
            return False  # continuous
        else:
            return True  # categorical

    def make_clickable(self, name):
        if name == "NA":
            return name
        else:
            link = "file:///" + os.getcwd() + "/hw4_output/figs/" + name + ".html"
            return '<a href="{}" target="_blank">{}</a>'.format(link, name)

    def make_report(self, dict):
        df = pd.DataFrame(dict)

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

        html = styler.to_html()

        with open("hw4_output/report.html", "w+") as file:
            file.write(html)
        file.close()

        filename = "file:///" + os.getcwd() + "/" + "hw4_output/report.html"
        webbrowser.open_new_tab(filename)

    def make_plots_rankings(self):
        self.response_type_check(self.df, self.response)

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
        for predictor in self.predictors:
            counter += 1

            out_dir_exist = os.path.exists("hw4_output/figs/{}".format(predictor))
            if out_dir_exist is False:
                os.makedirs("hw4_output/figs/{}".format(predictor))

            response_col.append(self.response)

            # Determine if the predictor is cat/cont
            cat_check = self.cat_cont_check(self.df, predictor)
            print(predictor)
            print(self.df[predictor].dtype)
            print(cat_check)

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
            if self.boolean_check is True and cat_check is True:
                self.plotter.bool_response_cat_predictor_plots(predictor)
                heatmap_col.append(f"{predictor}_heatmap")
                distribution_plot_col.append("NA")
                violin_plot_col.append("NA")
                scatter_plot_col.append("NA")
                t_score_col.append("NA")
                p_value_col.append("NA")
                logistic_regression_col.append("NA")
                linear_regression_col.append("NA")
            elif self.boolean_check is True and cat_check is False:
                self.plotter.bool_response_cont_predictor_plots(predictor)
                heatmap_col.append("NA")
                distribution_plot_col.append(f"{predictor}_distribution_plot")
                violin_plot_col.append(f"{predictor}_violin_plot")
                scatter_plot_col.append("NA")
                t_score, p_value = self.scorer.logistic_regression(predictor)
                t_score_col.append(t_score)
                p_value_col.append(p_value)
                logistic_regression_col.append(f"{predictor}_logistic_regression")
                linear_regression_col.append("NA")
            elif self.boolean_check is False and cat_check is True:
                self.plotter.cont_response_cat_predictor_plots(predictor)
                heatmap_col.append("NA")
                distribution_plot_col.append(f"{predictor}_distribution_plot")
                violin_plot_col.append(f"{predictor}_violin_plot")
                scatter_plot_col.append("NA")
                t_score_col.append("NA")
                p_value_col.append("NA")
                logistic_regression_col.append("NA")
                linear_regression_col.append("NA")
            else:
                self.plotter.cont_response_cont_predictor_plots(predictor)
                heatmap_col.append("NA")
                distribution_plot_col.append("NA")
                violin_plot_col.append("NA")
                scatter_plot_col.append(f"{predictor}_scatter_plot")
                t_score, p_value = self.scorer.linear_regression(predictor)
                t_score_col.append(t_score)
                p_value_col.append(p_value)
                logistic_regression_col.append("NA")
                linear_regression_col.append(f"{predictor}_linear_regression")

            # Difference with mean of response along with it's plot (weighted and unweighted)
            unweighted, weighted = self.scorer.diff_with_mean_of_resp(
                predictor, cat_check, self.boolean_check
            )
            mwr_unweighted_col.append(unweighted)
            mwr_weighted_col.append(weighted)
            mwr_plot_col.append(f"{predictor}_diff_w_mean_of_resp")
            rf_var_imp_col.append("NA")

        # Random Forest Variable importance ranking (continuous predictors only)
        importance = self.scorer.random_forest(cont_list, self.boolean_check)

        counter = -1

        for i in counter_list:
            counter += 1
            rf_var_imp_col[i] = importance[counter]

        data = {
            "Response ({})".format(self.response_name): response_col,
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

        self.make_report(data)
