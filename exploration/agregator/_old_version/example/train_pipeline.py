from glob import glob
import pickle
import numpy as np
import pandas as pd
from evraz.datascience import (EDA, REGRESSION_MODELS, CLASSIFICATION_MODELS, TASK_TYPES, Experiment,
                               ModelTrainer, SPLITTERS)
import os
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from tqdm import tqdm

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
MLFLOW_URL = os.getenv("MLFLOW_URL")
PROJECT_NAME = os.getenv("PROJECT_NAME")
COMMIT_HASH = os.getenv("COMMIT_HASH")

model_params = {
    "CatBoost": {
        'iterations': 5000,
        'depth': 8
    },
    "Linear": {},
}


def get_features(corr_matrix, target, target_min=0.1, corr_max=0.7):
    corr_matrix = corr_matrix[(corr_matrix[target] > target_min) |
                              (corr_matrix[target] < -target_min)]
    corr_matrix = corr_matrix[list(corr_matrix.index)]
    columns = corr_matrix.columns
    for i, column in enumerate(columns):
        for j in range(len(columns)):
            if abs(corr_matrix[column]
                   [j]) > corr_max and corr_matrix[column][j] != 1:
                if corr_matrix[target][j] < corr_matrix[target][i]:
                    corr_matrix[columns[j]] = 0
                else:
                    corr_matrix[columns[i]] = 0
    return list(
        (corr_matrix[[col for col in columns
                      if 1 in list(corr_matrix[col])]]).columns)


def FeatureEngineering(rails_inputs_and_outputs, *args, **kwargs):
    return rails_inputs_and_outputs.ffill().bfill(), None


def get_use_columns(data):
    use_columns = []
    for col in data.columns:
        if "billet_id" not in col and "LNK100" not in col:
            if len(data[col].dropna()) > 0 and data[col].std() > 0.00001:
                if len(data[col]) / len(data[col].dropna()) < 1.15:
                    use_columns.append(col)
    return use_columns


def get_billets_rolling_time(path_to_metadata):
    billets_info = list()
    metadata_files = glob(path_to_metadata)

    for metadata_path in metadata_files:
        metadata = pd.read_excel(metadata_path, engine='openpyxl')
        metadata["billet_id"] = "Л2" + metadata["Плавка"].str.split("-", expand=True).iloc[:, -1] + \
                                             metadata["Руч"].astype(str) + "0" + metadata["Заг"].astype(str)
        billets_info.append(metadata[["Вр.проката", "billet_id"]].dropna())
    billets_metadata = pd.concat(billets_info)

    return billets_metadata.drop_duplicates()

ABSPATH = r"Z:\DS\filtered_std_new\2023-03-27_2023-11-08"
NUM_OF_CORES = 19
path_to_files = os.path.join(ABSPATH, "*csv")
path_to_result = os.path.join(ABSPATH, "concated")


def get_model(i, target, path):
    with open(os.path.join(path_to_result, f"otk_df.pkl"), "rb") as handle:
        data_df = pickle.load(handle)
    with open(os.path.join(path_to_result, f"balanced_otk_importances.pkl"), "rb") as handle:
        imp = pickle.load(handle)

    vals = []
    for val in zip(data_df["A"], data_df["B"], data_df["C"], data_df["Y"]):
        if sum(val) > 0:
            vals.append(1)
        else:
            vals.append(0)
    data_df["ABCY"] = vals

    vals = []
    for val in zip(data_df["A"], data_df["Y"]):
        if sum(val) > 0:
            vals.append(1)
        else:
            vals.append(0)
    data_df["AY"] = vals

    for target in ["AY", "ABCY", "A", "B", "C", "Y"]:
        for exp_type in ["balanced"]:
            small_ft = dict(list(imp[target].items())[:50])
            features = [ft for ft, val in small_ft.items() if val > 0]
            train_data = data_df[[*features, target]]
            filtered_features = get_features(train_data.corr(), target, target_min=0, corr_max=0.6)
            train_data = train_data[filtered_features]
            if exp_type is "balanced":
                class_one = train_data[train_data[target] == 1].sample(frac=1).reset_index(drop=True)
                class_zero = train_data[train_data[target] == 0].sample(frac=1).reset_index(drop=True)
                min_size = min(len(class_one), len(class_zero))
                train_data = pd.concat([class_zero.iloc[:min_size, ],
                                        class_one.iloc[:min_size, ]])
            experiment = Experiment(
                EDA.Correlations,
                EDA.Histograms,
                EDA.DescribeFeatures,
                FeatureEngineering,
                ModelTrainer.TrainModel,
            )

            model_types = [CLASSIFICATION_MODELS.CatBoost]
            task_type = TASK_TYPES.CLASSIFICATION

            df = train_data.sample(frac=1).reset_index(drop=True)
            run_name = f"{target}_OTK(50Auto)_{exp_type}"
            for model_type in model_types:
                experiment.start(df,
                                 task_type=task_type,
                                 target=target,
                                 model_type=model_type,
                                 model_params=model_params[model_type.value],
                                 run_name=run_name,
                                 splitter=SPLITTERS.StratifiedKFold)
            print(f"Finished: {run_name}")


if __name__ == "__main__":
    get_model(None, None, None)
    # for metadata in [[0, 'LNK100_Torsion_[i_103.0]_[0_3]_[min]',
    #                   r'Z:\\DS\\filtered_std_new\\2023-03-27_2023-11-08\\concated\\best_df.pkl']]:
    #     get_model(metadata[0], metadata[1], metadata[2])