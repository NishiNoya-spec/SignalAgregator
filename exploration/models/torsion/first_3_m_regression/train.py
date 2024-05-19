from glob import glob

import pandas as pd

from exploration.models.otk import constants
from exploration.models.torsion.train_base import Trainer
from exploration.utils.model_training import get_low_corr_features

TARTGET = r""
PATH_TO_DATA = r""

MODEL_PARAMS = {
    "Linear": {},
    "CatBoost": {
        "iterations": 10000,
        "early_stopping_rounds": 10,
        "learning_rate": 0.1,
        "random_seed": 0,
        "depth": 4,
        'class_weight': 'balanced',
    },
    "RandomForest": {
        "random_seed": 0,
        "max_depth": 4,
    },
}

if __name__ == '__main__':
    data = pd.concat(
        [pd.read_csv(file_name) for file_name in glob(PATH_TO_DATA)]
    )
    data = data.drop_duplicates(subset=[constants.BILLET_ID])

    features = list(
        data.isna().sum()[data.isna().sum() / len(data) < 0.05].index
    )

    features = get_low_corr_features(data, features)

    data = data.dropna(subset=features)
    trainer = Trainer(
        data,
        features,
        TARTGET,
        MODEL_PARAMS,
    )
    trainer.train_regression(
        features,
        True,
    )
