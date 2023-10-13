from __future__ import division, print_function

import os

import warnings
import joblib
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from processing import processing

load_dotenv()
cudnn.benchmark = True
plt.ion()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def warn(*args, **kwargs):
    pass
warnings.warn = warn

GB_DIR = "./models/machine_learning/GradientBoost_model.pkl"
NN_DIR = "./sebas/tests/try_2/resnet_18_finetuning.pth"

class MergeModel:
    def __init__(self, base_model, nn_model, nn_weight=0.5, gb_weight=0.5, print_matrix=True) -> None:
        self.base_model = base_model
        self.nn_model = nn_model
        self.input_size = 224
        self.transformations = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.id_data, self.label = self.prepare_data()
        self.nn_model = self.load_Resnet_model()
        self.gb_model = self.load_GB_model()
        self.dataloader = self.prepare_loader()
        self.nn_weight = nn_weight
        self.gb_weight = gb_weight
        self.print_matrix = print_matrix

    def load_GB_model(self):
        GB_model = joblib.load(self.base_model)
        return GB_model[1]

    def load_Resnet_model(self):
        model = models.resnet18()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        model.load_state_dict(torch.load(self.nn_model))
        model.eval()
        return model

    def prepare_loader(self):
        dataset = datasets.ImageFolder("./data/MergeModel_Dataset", self.transformations)
        dataloader = DataLoader(dataset, batch_size=len(self.label))

        return dataloader

    def prepare_data(self):
        obj = processing(os.getenv("ROOT_MERGE"))
        X_train, _, Y_train, _ = train_test_split(obj.id_data, obj.label, test_size=1, shuffle=False)
        # print(Y_test)
        return X_train.to_numpy(), Y_train.to_numpy()

    def get_image(self, image_name_complete):
        image_name = image_name_complete.split("_")[0]
        directory = "./data/MergeModel_Dataset"
        for folder in os.listdir(directory):
            for filename in os.listdir(f"{directory}/{folder}"):
                if filename.startswith(image_name):
                    # image_path = os.path.join(f'{directory}{folder}/{filename}', image_name)
                    image_path = f"{directory}/{folder}/{filename}"
                    image = Image.open(image_path)
                    transformed_image = self.transformations(image)
                    input_batch = transformed_image.unsqueeze(0)
                    return input_batch
        return None

    def make_prediction(self, patient_data, image_name):
        image = self.get_image(image_name)
        nn_output = None
        if image is not None:
            # with torch.no_grad():
            nn_output = self.nn_model(image)
            nn_output = torch.argmax(nn_output)
            nn_output = (
                nn_output - 1
            ) ** 2  # esto se tuvo que hacer por que las lables estan volteadas cuando se entrendo el modelo de redes

        gb_output = self.gb_model.predict([patient_data])

        return nn_output, gb_output

    def run_model(self, nn_weight, gb_weight):
        results = []
        labels = []
        for i, data in enumerate(self.id_data):
            nn_output, gb_output = self.make_prediction(data[1:], data[0])
            if nn_output is None:
                print(f"{data[0]} x-ray was not found, skipping\n")
                continue
            labels.append(self.label[i])
            result = int(nn_output.numpy()) * nn_weight + int(gb_output) * gb_weight
            if result > 0.5:
                results.append(1)
            else:
                results.append(0)
        return results, labels

    def eval_model(self):
        predictions, labels = self.run_model(nn_weight=self.nn_weight, gb_weight=self.gb_weight)
        accuracy = accuracy_score(labels, predictions)
        presicion = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1_score_res = f1_score(labels, predictions)
        cm = confusion_matrix(labels, predictions)
        classification_report(labels, predictions)

        print(f"Accuracy: {accuracy:.3f}\nPresicion: {presicion:.3f}\nRecall: {recall:.3f}\nf1_score: {f1_score_res:.3f}\n")
        print(f"{cm}\n")
        if self.print_matrix:
            self.create_cm(cm, "MergeModel")

    def create_cm(self, cm, model_name):
        x = ["No COPD", "COPD"]
        y = ["No COPD", "COPD"]

        z_text = [[str(y) for y in x] for x in cm]

        fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=z_text, colorscale="magenta")
        fig.update_layout(title_text=f"<i><b>Confusion matrix {model_name} </b></i>")

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

if __name__ == "__main__":
    model = MergeModel(GB_DIR, NN_DIR, 0.25, 0.75)
    model.eval_model()
