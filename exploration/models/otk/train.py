from dataclasses import dataclass
from glob import glob
from typing import List

import constants
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from evraz.datascience import (
    CLASSIFICATION_MODELS,
    EDA,
    REGRESSION_MODELS,
    TASK_TYPES,
    Experiment,
    ModelTrainer,
)

from exploration.utils.model_training import (
    get_low_corr_features,
    select_features_syntetic,
)

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
CRATE = "Y"
MIN_YEAR = "2023"
DEFECT_CRATES = [CRATE, "A;B;C;Y"]
PATH_TO_DATA = r"\\ZSMK-9684-001\Data\DS\ИТС + Передел\results\20*.csv"


@dataclass
class Trainer:
    data: pd.DataFrame
    features: List[str]

    def train_classification(self, features, shufl_data=False):
        task_type = TASK_TYPES.CLASSIFICATION
        model_type = CLASSIFICATION_MODELS.CatBoost
        self._train(features, task_type, model_type, shufl_data=shufl_data)

    def train_regression(self, features, shufl_data=False):
        task_type = TASK_TYPES.REGRESSION
        model_type = REGRESSION_MODELS.CatBoost
        self._train(features, task_type, model_type, shufl_data=shufl_data)

    def _train(self, features, task_type, model_type, shufl_data):
        train_data = self.data.copy()    # .reset_index()
        if shufl_data:
            undersampler = RandomUnderSampler(sampling_strategy=0.5)
            X, y = undersampler.fit_resample(
                train_data.drop([constants.DEFECT_COL], axis=1),
                train_data[constants.DEFECT_COL]
            )
            X[constants.DEFECT_COL] = y
            train_data = X    # .set_index("moment")

        features = select_features_syntetic(
            train_data,
            features,
        )

        self.features = features['selected_features_names']
        self.features = list(set(self.features + [f"dl{CRATE}", "марка"]))
        experiment = Experiment(
            EDA.Correlations,
            EDA.Histograms,
            EDA.DescribeFeatures,
            self.FeatureEngineering,
            ModelTrainer.TrainModel,
        )

        experiment.start(
            train_data.sample(frac=1)[self.features + [constants.DEFECT_COL]],
            task_type=task_type,
            target=constants.DEFECT_COL,
            model_type=model_type,
            model_params=MODEL_PARAMS[model_type.value],
            run_name=constants.DEFECT_COL,
            min_periods=10
        )

    @staticmethod
    def FeatureEngineering(rails_inputs_and_outputs, *args, **kwargs):
        return rails_inputs_and_outputs.ffill().bfill(), None


if __name__ == '__main__':

    data = []

    for file_name in glob(
            r"\\ZSMK-9684-001\Data\DS\ИТС + Передел\results\20*.csv"):
        df = pd.read_csv(file_name).set_index("moment").drop(
            columns=['Unnamed: 0', 'Unnamed: 0.1']
        )
        df = df[(df[constants.DEFECT_CRATE].isin(DEFECT_CRATES))
                & (df["марка"].isin(['Э76ХФ', 'Э90ХАФ', '76ХФ', '90ХАФ']))]
        df["марка"] = df["марка"].map(
            {
                'Э76ХФ': 0,
                'Э90ХАФ': 1,
                '76ХФ': 0,
                '90ХАФ': 1
            }
        )
        data.append(df)

    data = pd.concat(data)
    data = data.drop_duplicates(subset=[constants.BILLET_ID])

    features = list(
        data.isna().sum()[data.isna().sum() / len(data) < 0.05].index
    )[3:]
    data = data[data.index.str[:4] >= MIN_YEAR]
    features = get_low_corr_features(data, features)

    data = data.dropna(subset=features)
    trainer = Trainer(data, features)
    trainer.train_classification(
        features,
        True,
    )
