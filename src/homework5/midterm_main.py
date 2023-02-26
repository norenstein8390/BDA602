import os
import webbrowser

import pandas as pd
from midterm_bruteforce import MidtermBruteForce
from midterm_correlations import MidtermCorrelations


class MidtermReportMaker:
    def __init__(self, df, predictors, response, version):
        self.df = df.dropna().reset_index()
        self.predictors = predictors
        self.response = response
        self.correlations = MidtermCorrelations(self.df, response)
        self.bruteforce = MidtermBruteForce(self.df, response)
        self.version = version

        out_dir_exist = os.path.exists("final/midterm_output/figs")
        if out_dir_exist is False:
            os.makedirs("final/midterm_output/figs")

    def make_clickable(self, name):
        # link = f"file:///{os.getcwd()}/final/midterm_output/figs/{name}.html"
        # link = f"file:///src/final/midterm_output/figs/{name}.html"
        link = f"midterm_output/figs/{name}.html"
        return f'<a href="{link}" target="_blank">{name}</a>'

    def make_report(self, html):
        with open("final/midterm_output/report.html", "w+") as file:
            file.write(html)
        file.close()

        filename = f"file:///{os.getcwd()}/final/midterm_output/report.html"
        webbrowser.open_new_tab(filename)

    def make_correlations_bruteforce(self):
        response_text = f"Response '{self.response}' is boolean"

        cont_predictor_start = 0

        cont_cont_correlation_data = {
            "Predictors (Cont/Cont)": [],
            "Pearson's r": [],
            "Absolute Value of Correlation": [],
            "Scatter Plot": [],
        }

        self.correlations.cont_cont_correlation(
            self.predictors, cont_predictor_start, cont_cont_correlation_data
        )
        cont_cont_correlation_df = pd.DataFrame(cont_cont_correlation_data)
        cont_cont_correlation_df = cont_cont_correlation_df.sort_values(
            by=["Absolute Value of Correlation"], ascending=False
        )
        cont_cont_styler = cont_cont_correlation_df.style.format(
            {"Scatter Plot": self.make_clickable}
        )

        matrix_data = {"Cont/Cont Correlation Matrix": []}

        self.correlations.cont_cont_matrix(
            self.predictors, cont_predictor_start, matrix_data, self.version
        )

        matrix_df = pd.DataFrame(matrix_data)

        matrix_styler = matrix_df.style.format(
            {"Cont/Cont Correlation Matrix": self.make_clickable}
        )

        cont_cont_brute_force_data = {
            "Predictors (Cont/Cont)": [],
            "Difference of Mean Response": [],
            "Weighted Difference of Mean Response": [],
            "Bin Mean Plot": [],
            "Bin Residual Plot": [],
        }

        self.bruteforce.cont_cont_brute_force(
            self.predictors, cont_predictor_start, cont_cont_brute_force_data
        )
        cont_cont_brute_force_df = pd.DataFrame(cont_cont_brute_force_data)
        cont_cont_brute_force_df = cont_cont_brute_force_df.sort_values(
            by=["Weighted Difference of Mean Response"], ascending=False
        )
        cont_cont_brute_force_styler = cont_cont_brute_force_df.style.format(
            {
                "Bin Mean Plot": self.make_clickable,
                "Bin Residual Plot": self.make_clickable,
            }
        )

        html = "<h2>Correlation</h2>\n\n"
        html += cont_cont_styler.to_html() + "\n\n"

        html += (
            "<h2>Correlation Matrices</h2>\n\n"
            + matrix_styler.to_html()
            + "\n\n<h2>Brute Force - {}</h2>\n\n".format(response_text)
        )

        html += cont_cont_brute_force_styler.to_html() + "\n\n"

        return html
