import numpy as np
from plotly import figure_factory as ff
from plotly import graph_objects as go


class Homework4Plotter:
    def __init__(self, df, response):
        self.df = df
        self.response = response

    def bool_response_cont_predictor_plots(self, predictor):
        # Distribution Plot
        responses = self.df[self.response].unique()
        response1 = responses[0]
        response2 = responses[1]

        response1_predictor = self.df[self.df[self.response] == response1][
            predictor
        ].dropna()
        response2_predictor = self.df[self.df[self.response] == response2][
            predictor
        ].dropna()

        hist_data = [response1_predictor, response2_predictor]
        group_labels = [
            "Response = {}".format(response1),
            "Response = {}".format(response2),
        ]

        fig1 = ff.create_distplot(hist_data, group_labels)
        fig1.update_layout(
            title="{} v. {}: Distribution Plot".format(predictor, self.response),
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
            title="{} vs. {}: Violin Plot".format(predictor, self.response),
            xaxis_title=self.response,
            yaxis_title=predictor,
        )
        fig1.write_html(f"homework5/hw4_output/figs/{predictor}_distribution_plot.html")
        fig2.write_html(f"homework5/hw4_output/figs/{predictor}_violin_plot.html")
