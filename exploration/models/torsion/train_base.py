from dataclasses import dataclass
from typing import List

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

from exploration.utils.model_training import select_features_syntetic


@dataclass
class Trainer:
    data: pd.DataFrame
    features: List[str]
    target: str
    model_params: dict

    def train_classification(self, features, shufl_data=False):
        task_type = TASK_TYPES.CLASSIFICATION
        model_type = CLASSIFICATION_MODELS.CatBoost
        self._train(features, task_type, model_type, shufl_data=shufl_data)

    def train_regression(self, features, shufl_data=False):
        task_type = TASK_TYPES.REGRESSION
        model_type = REGRESSION_MODELS.CatBoost
        self._train(features, task_type, model_type, shufl_data=shufl_data)

    def _train(self, features, task_type, model_type, shufl_data):
        train_data = self.data.copy()
        if shufl_data:
            undersampler = RandomUnderSampler(sampling_strategy=0.5)
            X, y = undersampler.fit_resample(
                train_data.drop([self.target], axis=1), train_data[self.target]
            )
            X[self.target] = y
            train_data = X    # .set_index("moment")

        features = select_features_syntetic(
            train_data,
            features,
        )

        self.features = features['selected_features_names']

        experiment = Experiment(
            EDA.Correlations,
            EDA.Histograms,
            EDA.DescribeFeatures,
            self.FeatureEngineering,
            ModelTrainer.TrainModel,
        )

        experiment.start(
            train_data.sample(frac=1)[self.features + [self.target]],
            task_type=task_type,
            target=self.target,
            model_type=model_type,
            model_params=self.model_params[model_type.value],
            run_name=self.target,
            min_periods=10
        )

    @staticmethod
    def FeatureEngineering(rails_inputs_and_outputs, *args, **kwargs):
        return rails_inputs_and_outputs.ffill().bfill(), None
