# Exploratory data analysis
import pandas as pd
import global_variables as gv
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo


class processing:
    def __init__(self, root) -> None:
        df = pd.read_excel(root)
        df = self.preprocess_dataset(df)
        df = df.convert_dtypes()
        #self.resume_data(df)
        self.complete_data = df
        self.label = df.epoc.astype(int)
        self.data = df[
            [
                "tbco",
                "wheeze",
                "cough",
                "dyspnea",
                "ysmk",
                "yr_qty",
                "pks",
                "height",
                "weight",
                "age",
                "bsa",
                "fvc",
                "fev1",
                "fev1_fvc",
                "fef25",
                "fef50",
                "fef75",
                "fef25_75",
                "fef_max",
                "fivc",
                "fif50",
                "fif_max",
            ]
        ]
        self.id_data = df[
            [
                "id",
                "tbco",
                "wheeze",
                "cough",
                "dyspnea",
                "ysmk",
                "yr_qty",
                "pks",
                "height",
                "weight",
                "age",
                "bsa",
                "fvc",
                "fev1",
                "fev1_fvc",
                "fef25",
                "fef50",
                "fef75",
                "fef25_75",
                "fef_max",
                "fivc",
                "fif50",
                "fif_max",
            ]
        ]

        #print(self.data.tbco.value_counts())

    def preprocess_dataset(self, df):
        fields = {
            "ysmk": 0,
            "pks": 0,
            "yr_qty": 0,
            "medication": "",
            "diagnosis": "",
            "dyspnea": "",
            "cough": "",
            "wheeze": "",
            "tbco": "",
            "conclusion": "",
            "height": "",
        }
        df.fillna(value=fields, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype("datetime64")
        df.rename(columns={"epoc ": "epoc"}, inplace=True)
        df["has_epoc"] = df["epoc"].apply(lambda x: "Yes" if x == 1 else "No")
        df["sex"] = df["sex"].apply(lambda x: "F" if x == "femenino" else "M")
        df["smoker"] = df["tbco"].apply(lambda x: "Smoker" if (x != "nunca fumo") and (x != "") else "No Smoker")
        df["dyspnea"].replace(gv.dyspnea_mapper, inplace=True)
        df["cough"].replace(gv.cough_mapper, inplace=True)
        df["wheeze"].replace(gv.wheeze_mapper, inplace=True)
        df["tbco"].replace(gv.tbco_mapper, inplace=True)
        return df

    def resume_data(self, df):
        print("----------------------------------------------------")
        print("Number of samples: ", len(df))
        print("----------------------------------------------------")
        print(df.describe())
        print("---------------------------------------------------------------------------------------------------")
        #self.plot_data(df, "Gender", "sex", "Smoker", "smoker", "COPD", "has_epoc")

        #fig = px.box(df, x="height", title="Height", width=800, height=400)
        #fig.show()
        # traces.append(go.Box(name='Age', x=df.age))
        # traces.append(go.Box(name='Height', x=df.height))
        # traces.append(go.Box(name='Weight', x=df.weight))

        # layout = go.Layout(boxmode='group')
        # fig = go.Figure(data=traces, layout=layout)
        # pyo.iplot(fig)
        return

    def plot_data(self, df, title1, variable1, title2, variable2, title3, variable3):
        cols = ("lightseagreen", "lightcoral")
        values1 = df[variable1].value_counts().values
        index1 = df[variable1].value_counts().index

        values2 = df[variable2].value_counts().values
        index2 = df[variable2].value_counts().index

        values3 = df[variable3].value_counts().values
        index3 = df[variable3].value_counts().index

        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
            subplot_titles=(title1, title2, title3),
        )

        fig.add_trace(go.Bar(x=index1, y=values1, marker=dict(color=cols)), row=1, col=1)
        fig.add_trace(go.Bar(x=index2, y=values2, marker=dict(color=cols)), row=1, col=2)
        fig.add_trace(go.Bar(x=index3, y=values3, marker=dict(color=cols)), row=1, col=3)
        fig.update_layout(height=700, showlegend=False, title_text="SUMMARY")
        fig.show()


if __name__ == "__main__":
    preprocess = processing("./data/spirometry.xlsx")
    # df = preprocess.preprocess_dataset(preprocess.complete_data)
    preprocess.complete_data.to_csv("./data/processed_data.csv", sep=",")
