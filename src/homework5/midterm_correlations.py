from plotly import express as px
from scipy.stats import pearsonr


class MidtermCorrelations:
    def __init__(self, df, response):
        self.df = df
        self.response = response

    def cont_cont_plots(self, x_predictor, y_predictor):
        fig = px.scatter(self.df, x=x_predictor, y=y_predictor, trendline="ols")
        fig.update_layout(
            title=f"{x_predictor} vs. {y_predictor}: Scatter Plot",
            xaxis_title=x_predictor,
            yaxis_title=y_predictor,
        )
        name = x_predictor + "_" + y_predictor + "_scatter_plot"
        link = "homework5/midterm_output/figs/" + name + ".html"
        fig.write_html(link)
        return name

    def cont_cont_correlation(self, predictors, cont_predictor_start, data):
        cont_predictors = predictors[cont_predictor_start:]

        for i in range(len(cont_predictors)):
            x_predictor = cont_predictors[i]

            for j in range(i + 1, len(cont_predictors)):
                y_predictor = cont_predictors[j]
                data["Predictors (Cont/Cont)"].append(
                    x_predictor + " and " + y_predictor
                )
                data["Pearson's r"].append(
                    pearsonr(self.df[x_predictor], self.df[y_predictor])[0]
                )
                data["Absolute Value of Correlation"].append(
                    abs(pearsonr(self.df[x_predictor], self.df[y_predictor])[0])
                )
                data["Scatter Plot"].append(
                    self.cont_cont_plots(x_predictor, y_predictor)
                )

    def cont_cont_matrix(self, predictors, cont_predictor_start, data):
        if cont_predictor_start == len(predictors):
            data["Cont/Cont Correlation Matrix"].append("N/A")
            return

        cont_predictors = predictors[cont_predictor_start:]
        matrix = []

        for i in range(len(cont_predictors)):
            cont_predictor_i = cont_predictors[i]
            row = []

            for j in range(len(cont_predictors)):
                cont_predictor_j = cont_predictors[j]
                row.append(
                    abs(
                        pearsonr(self.df[cont_predictor_i], self.df[cont_predictor_j])[
                            0
                        ]
                    )
                )

            matrix.append(row)

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
        link = f"homework5/midterm_output/figs/{name}.html"
        fig.write_html(link)
        data["Cont/Cont Correlation Matrix"].append(name)
