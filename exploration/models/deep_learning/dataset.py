import json
from glob import glob
from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from exploration.models.deep_learning.constants import (
    INPUT_SIZE,
    MAX_VALUES,
    MIN_VALUES,
    TARGET,
    USE_COLS,
)


class BilletsDataset(Dataset):

    def __init__(self, files_paths: List[str]):
        self.files_paths = files_paths

    def __len__(self):
        return len(self.files_paths)

    def __getitem__(self, idx):
        data = pd.read_csv(
            self.files_paths[idx], usecols=USE_COLS, encoding="utf-8"
        )
        data[TARGET] = data[TARGET].abs()
        data = data.sub(MIN_VALUES).div(MAX_VALUES)
        # data = data.div(MAX_VALUES)

        target = data[TARGET]
        # target = (data[TARGET].copy().abs() > 400).astype(int)

        # data = data.sub(MEAN_VALUES).div(STD_VALUES) # поделить на максимум

        features = torch.tensor(data.drop([TARGET], axis=1).values).float()
        inputs, targets = torch.transpose(features, 0,
                                          1), torch.tensor(target).float()

        padded_inputs = torch.zeros((INPUT_SIZE, 832))
        padded_target = torch.zeros((1, 832))
        padded_inputs[:, 4:829] = inputs
        padded_target[:, 4:829] = targets
        return padded_inputs, padded_target


class BilletsJsonDataset(Dataset):

    def __init__(self, files_paths: List[str]):
        self.files_paths = files_paths

    def __len__(self):
        return len(self.files_paths)

    def __getitem__(self, idx):
        with open(self.files_paths[idx], "r") as read_file:
            data = json.load(read_file)
        return torch.tensor(data["features"]
                            ).float(), torch.tensor(data["target"]).float()


PATH_TO_DATA = r"E:\zharikova_ep\data_9684\results\data_for_dl\train\*.csv"
if __name__ == "__main__":
    billets_dataset = BilletsDataset(glob(PATH_TO_DATA))
    max_values = {col: -1000 for col in USE_COLS}
    min_values = {col: 1000 for col in USE_COLS}
    for path in tqdm(glob(PATH_TO_DATA)):
        data = pd.read_csv(path, usecols=USE_COLS, encoding="utf-8").iloc[7:-7]
        data[TARGET] = data[TARGET].abs()
        data_min = data.min().to_dict()
        data_max = data.max().to_dict()
        for key, value in data_min.items():
            max_values[key] = max([data_max[key], max_values[key]])
            min_values[key] = min([value, min_values[key]])

    print(min_values)
    print()
    print(max_values)
