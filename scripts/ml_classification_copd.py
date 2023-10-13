import pickle
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.offline as pyo
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


class ml_classification_copd:
    def __init__(self, data, label, split=True, validation=None, Y_validation=None) -> None:
        self.seed = 7
        self.params_grid = {
            "KNN": {
                'n_neighbors': np.arange(5,12,1),
                'weights': ['uniform', 'distance']    
            },
            "NB": {
                "var_smoothing": np.arange(0.1, 0.2, 0.01),    
            }, 
            "SVM": {
                'C': np.arange(1,6,1),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': np.arange(3,6,1),
                'gamma': np.arange(0.01, 0.1, 0.01) 
            }, 
            "DesicionTree": {
                "criterion": ['gini', 'entropy', 'log_loss'],
                'splitter' : ['best', 'random'],
                'max_depth': np.arange(5,12,1),
                'min_samples_split': np.arange(2,7,1),
                'min_samples_leaf': np.arange(1,5,1),
                'max_features': ['sqrt','log2', None]
            },
            "RandomForest": {
                'n_estimators': np.arange(100,200,50),
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_depth': np.arange(50,90,10),
                'min_samples_split': np.arange(5, 8, 1),
                'min_samples_leaf': np.arange(4, 7, 1),
                'max_features': ['sqrt', 'log2', None]
            }, 
            "GradientBoost": {
                'loss': ['log_loss','exponential'],
                'n_estimators': np.arange(100,200,50),
                'criterion': ['frifriedman_mse', 'squared_error'],
                'max_depth': np.arange(50,90,10),
                'min_samples_split': np.arange(5,8,1),
                'min_samples_leaf': np.arange(3, 7, 1),
            }, 
        }

        if split:
            self.validation_size = 0.30
            copd = label[label == 1].index
            no_copd = label[label == 0].index
            data_copd = data.iloc[copd]
            data_no_copd = data.iloc[no_copd]
            len(data)
            df_minority_upsampled = resample(
                data_no_copd,
                replace=True,  # sample with replacement
                n_samples=143,  # to match majority class
                random_state=123,
            )  # reproducible results
            final_data = pd.concat([data_copd, df_minority_upsampled])
            x = np.zeros(143)
            final_label = pd.concat([label[label == 1], pd.Series(x)])
            final_label = final_label.astype(int)
            X_train, X_validation, self.Y_train, self.Y_validation = model_selection.train_test_split(
                final_data, final_label, test_size=self.validation_size, random_state=123
            )
        else:
            X_train = data
            self.Y_train = label
            X_validation = validation
            self.Y_validation = Y_validation

        self.X_train = X_train.astype(float)
        self.X_validation = X_validation.astype(float)
        print("-------------TRAIN----------------")
        print(len(self.X_train))
        print("-------------TEST-----------------")
        print(len(self.X_validation))
        models = self.train_models()
        for model in models:
            # filename = f'{model[0]}_model.sav'
            # pickle.dump(model[1], open(f'./models/machine_learning/{filename}', 'wb'))
            filename = f"{model[0]}_model.pkl"
            joblib.dump(model, f"./models/machine_learning/{filename}")

    def train_models(self):
        scoring = "accuracy"
        models = []
        models.append(("KNN", KNeighborsClassifier()))
        models.append(("NB", GaussianNB()))
        models.append(("SVM", SVC()))
        models.append(("DesicionTree", DecisionTreeClassifier()))
        models.append(("RandomForest", RandomForestClassifier()))
        models.append(("GradientBoost", GradientBoostingClassifier()))

        best_params = {}
        names = []
        for name, model in models:
            names.append(name)
            kfold = model_selection.KFold(n_splits=10, random_state=self.seed, shuffle=True)
            rf_cv = GridSearchCV(estimator=model, param_grid=self.params_grid[name], cv=kfold, scoring=scoring, n_jobs=6)
            rf_cv.fit(self.X_train, self.Y_train)

            best_params[name] = rf_cv.best_params_

            print(f"For model {name}, the best parameters were {best_params[name]}\n")

            model = model.set_params(**best_params[name])
            model.fit(self.X_train, self.Y_train)

        self.plot_best_models(best_params, models)
        return models

    def plot_results(self, names, results):
        traces = []
        for i, data in enumerate(results):
            traces.append(go.Box(name=names[i], x=data))
        layout = go.Layout(boxmode="group", width=800, height=400)
        fig = go.Figure(data=traces, layout=layout)
        pyo.iplot(fig)

    def plot_best_models(self, best_params, models):
        x = ["No COPD", "COPD"]
        y = ["No COPD", "COPD"]

        model_results = {}
        for name, model in models:
            print("Model: ", name)
            best_model = model.set_params(**best_params[name])
            best_model.fit(self.X_train, self.Y_train)

            predictions = best_model.predict(self.X_validation)

            accuracy = accuracy_score(self.Y_validation, predictions)
            presicion = precision_score(self.Y_validation, predictions)
            recall = recall_score(self.Y_validation, predictions)
            f1_score_res = f1_score(self.Y_validation, predictions)
            cm = confusion_matrix(self.Y_validation, predictions)
            classification_report(self.Y_validation, predictions)

            model_results[name] = [accuracy, presicion, recall, f1_score_res]

            print(f"Accuracy: {accuracy}\nPresicion: {presicion}\nRecall: {recall}\nf1_score: {f1_score_res}\n")
            print(f'{cm}\n')
            if name == 'RandomForest' or name == 'GradientBoost':
                print(model.feature_importances_) 

            z_text = [[str(y) for y in x] for x in cm]

            fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=z_text, colorscale="magenta")
            fig.update_layout(title_text=f"<i><b>Confusion matrix {name}</b></i>")

            fig.add_annotation(
                dict(
                    font=dict(color="black", size=20),
                    x=0.5,
                    y=-0.15,
                    showarrow=False,
                    text="Predicted value",
                    xref="paper",
                    yref="paper",
                )
            )
            fig.add_annotation(
                dict(
                    font=dict(color="black", size=20),
                    x=-0.35,
                    y=0.5,
                    showarrow=False,
                    text="Real value",
                    textangle=-90,
                    xref="paper",
                    yref="paper",
                )
            )
            fig.update_layout(width=800, height=400)
            fig.show()

        results_df = pd.DataFrame.from_dict(
            model_results, orient="index", columns=["Accuracy", "Presicion", "Recall", "F1_Score"]
        )
        results_df.to_excel("base_models_results.xlsx")
