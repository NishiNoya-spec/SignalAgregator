from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.metrics import (
    max_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from tqdm import tqdm

from exploration.models.deep_learning.constants import (
    INPUT_SIZE,
    MAX_VALUES,
    MIN_VALUES,
    OUTPUT_SIZE,
    PATH_TO_TEST_DATA,
    PATH_TO_TRAIN_DATA,
    PATH_TO_VAL_DATA,
    STEP,
)
from exploration.models.deep_learning.dataset import BilletsDataset
from exploration.models.deep_learning.metrics import Metrics
from exploration.models.deep_learning.model import TorsionUNet
from exploration.models.deep_learning.train import get_iou

PATH_TO_MODEL = r"best.pth"


def denormalize_data(data: np.ndarray):
    return (data * MAX_VALUES['LNK100_Torsion'] + MIN_VALUES['LNK100_Torsion'])


def draw_scatter_plot(prediction, target, save_path):
    scatter = px.scatter(x=prediction, y=target)
    scatter.update_layout(xaxis_title="Predict", yaxis_title="Test")
    # add 45 line for scatter
    min_45_line = np.min([prediction, target])
    max_45_line = np.max([prediction, target])
    line_45 = px.line(
        x=[min_45_line, max_45_line], y=[min_45_line, max_45_line]
    )
    line_45.update_traces(line_color='#A0A0A0')
    scatter.add_trace(line_45.data[0])

    scatter.write_image(save_path)


def draw_target_prediction_plot(prediction, target, save_path):
    plt.plot(target, "r", prediction, "g")
    plt.savefig(save_path)


def draw_diff_target_prediction(prediction, target, save_path):
    plt.plot(np.abs(target - prediction))
    plt.savefig(save_path)


def get_metrics(prediction, target, defect_prediction, defect_target, name):
    metrics_calculator = Metrics(defect_prediction, defect_target)

    return {
        'point': name,
        'r2': r2_score(prediction, target),
        'mse': mean_squared_error(prediction, target),
        'mape': mean_absolute_percentage_error(prediction, target),
        'max_error': max_error(prediction, target),
        'precision': metrics_calculator.precision,
        'recall': metrics_calculator.recall,
        'f1-score': metrics_calculator.f1_score,
        'accuracy': metrics_calculator.accuracy,
        'iou': get_iou(defect_prediction, defect_target),
    }


def report_for_every_n_metres(prediction, target, n_meters, billet_id):
    num_steps = int(n_meters / STEP)
    points = [i // num_steps for i in range(len(prediction))]
    result = pd.DataFrame(
        {
            "points": points,
            "prediction": np.abs(prediction),
            "target": np.abs(target)
        }
    )
    report = result.groupby("points").max()
    report["point"] = [(i + 1) * n_meters for i in range(len(report))]
    report['BilletId'] = billet_id
    return report[['BilletId', "point", "target", "prediction"]]


if __name__ == "__main__":

    print("Prepared dataset")
    # threshold = 0.5
    cols_to_write = [
        'target',
        'prediction',
        'diff',
        'iou',
        'r2',
        'mse',
        'mape',
        'max_error',
    ]
    test_datasets = [
        BilletsDataset(glob(PATH_TO_TEST_DATA)),
        BilletsDataset(glob(PATH_TO_TRAIN_DATA)),
        BilletsDataset(glob(PATH_TO_VAL_DATA))
    ]
    model = TorsionUNet(INPUT_SIZE, OUTPUT_SIZE)
    model.load_state_dict(torch.load(PATH_TO_MODEL, ))
    model.to("cuda")
    model.eval()
    predictions = []
    targets = []

    defect_predictions = []
    defect_targets = []
    metrics = list()
    report = []
    for test_dataset in tqdm(test_datasets):
        for i in tqdm(range(len(test_dataset))):
            features, target = test_dataset[i]
            features = features.to("cuda")
            prediction = denormalize_data(
                model(features.unsqueeze(0))[0, 0, 4:-3].detach().cpu().numpy()
            )[:int(6 / 0.125)]
            target = denormalize_data(target[0, 4:-3].detach().cpu().numpy()
                                      )[:int(6 / 0.125)]

            defect_predictions.append((np.abs(prediction) >= 500).astype(int))
            defect_targets.append((np.abs(target) >= 500).astype(int))

            predictions.append(prediction)
            targets.append(target)

            image_name = test_dataset.files_paths[i].split("/")[-1][:-4]

            report.append(
                report_for_every_n_metres(prediction, target, 3, image_name)
            )

    pd.concat(report).reset_index(drop=True).to_csv(
        "results_test/report.csv", sep=";", decimal=","
    )
    pd.DataFrame(metrics).to_csv(
        "results_test/rail_metrics.csv", sep=";", decimal=","
    )

    targets = np.transpose(np.array(targets))
    predictions = np.transpose(np.array(predictions))
    defect_predictions = np.transpose(np.array(defect_predictions))
    defect_targets = np.transpose(np.array(defect_targets))

    metrics = list()
    for i in range(len(predictions)):
        fig = plt.figure()
        name = f"{STEP * i}m"
        target = targets[i]
        prediction = predictions[i]

        defect_target = defect_targets[i]
        defect_prediction = defect_predictions[i]

        image_name = test_dataset.files_paths[i].split("/")[-1][:-4]

        metrics.append(
            get_metrics(
                prediction, target, defect_prediction, defect_target, name
            )
        )

        draw_target_prediction_plot(
            prediction, target, f'results_test/point_diagrams/{name}.png'
        )
        draw_diff_target_prediction(
            prediction, target, f'results_test/point_diagrams/{name}_diff.png'
        )
        draw_scatter_plot(
            prediction, target,
            f'results_test/point_diagrams/{name}_scatter.png'
        )

        target_vs_prediction = pd.DataFrame(
            {
                'target': target,
                'prediction': prediction,
                'diff': np.abs(target - prediction),
                'defect_target': defect_target,
                'defect_prediction': defect_prediction,
            }
        )

        target_vs_prediction.to_csv(
            f"results_test/{name}.csv", sep=";", decimal=","
        )
    pd.DataFrame(metrics).to_csv(
        "results_test/metrics.csv", sep=";", decimal=","
    )
