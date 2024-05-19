import os
from datetime import datetime, timedelta
from glob import glob

import constants
import pandas as pd
import shap
from catboost import CatBoostClassifier

from exploration.utils.metadata import MethaDataProcessor

CRATE = "Y"
PATH_TO_MODEL = r"path\to\model.cb"
PATH_TO_FEATURES = r"\\ZSMK-9684-001\Data\DS\ИТС + Передел\results\202*.csv"
PATH_TO_METADATA = r"\\ZSMK-9684-001\Data\DS\metadata\*.xlsx"
SAVE_PATH = ""


class Model:

    def __init__(self, path_to_model: str, path_to_features: str):
        self.model = self._get_model(path_to_model)
        self.explainer = self._get_shap_explainer()

        self.dataset = self._load_input_features(path_to_features)

    def _load_input_features(self, path_to_features):
        data = []
        for path in glob(path_to_features):
            data.append(
                pd.read_csv(
                    path,
                    usecols=[
                        constants.DEFECT_CRATE, constants.BILLET_ID,
                        constants.DEFECT_COL
                    ] + self.model.feature_names_
                )
            )

        data = pd.concat(data).set_index(constants.BILLET_ID)

        crate_filter = data[constants.DEFECT_CRATE].isin([CRATE, "A;B;C;Y"])
        mark_filter = data["марка"].isin(['Э76ХФ', 'Э90ХАФ', '76ХФ', '90ХАФ'])
        data = data[(crate_filter
                     & mark_filter)].drop(columns=[constants.DEFECT_CRATE])

        data["марка"] = data["марка"].map(
            {
                'Э76ХФ': 0,
                'Э90ХАФ': 1,
                '76ХФ': 0,
                '90ХАФ': 1
            }
        )

        methadata_processor = MethaDataProcessor(PATH_TO_METADATA)
        time_data = methadata_processor.get_billets_rolling_time()
        data = pd.merge(time_data, data, left_index=True, right_index=True)

        return data.drop_duplicates(
            subset=constants.MOMENT, keep='last'
        ).set_index(constants.MOMENT).fillna(0)

    def _get_model(self, path_to_model):
        return CatBoostClassifier().load_model(path_to_model)

    def _get_shap_explainer(self):
        return shap.Explainer(self.model)

    def _get_impact(self, input_features):
        shap_values = pd.DataFrame(self.explainer.shap_values(input_features))
        shap_values.columns = [
            "impact_" + col for col in input_features.columns
        ]
        return shap_values


class ModelTester(Model):

    def _predict(self):
        input_features = self.dataset.drop(columns=[constants.DEFECT_COL])
        indexes = input_features.index

        impact = self._get_impact(input_features)
        impact.index = indexes

        result_table = pd.concat([input_features, impact], axis=1)
        result_table = (
            result_table[sum(
                [[col, "impact_" + col] for col in input_features.columns], []
            )]
        )

        prediction = pd.DataFrame(self.model.predict_proba(input_features))
        prediction.columns = ["_", "prediction"]
        prediction.index = indexes
        prediction = \
            pd.merge(self.dataset, prediction, left_index=True,
                     right_index=True)[
                [constants.DEFECT_COL, "prediction"]]
        return pd.concat((prediction, result_table), axis=1)

    def test(self):
        return self._predict()


class CutTester(Model):

    def _predict(self, date: datetime):
        input_features = self.dataset[
            (self.dataset.index >= date)
            & (self.dataset.index < date + timedelta(seconds=1))].sort_index()
        fact = input_features[constants.DEFECT_COL].copy().reset_index(
            drop=True
        )
        input_features = input_features.drop(columns=[constants.DEFECT_COL]
                                             ).reset_index(drop=True)
        result_table = []
        for i in range(30):
            input_features[f"dl{CRATE}"] = 750 + 50 * i
            impact = self._get_impact(input_features)
            curr_table = pd.concat([input_features, impact], axis=1)
            prediction = pd.DataFrame(self.model.predict_proba(input_features))
            prediction.columns = ["_", "prediction"]
            curr_table["prediction"] = prediction["prediction"]
            curr_table["fact"] = fact
            curr_table["moment"] = date
            result_table.append(
                curr_table[[
                    "moment",
                    "fact",
                    "prediction",
                ] + sum(
                    [[col, "impact_" + col]
                     for col in input_features.columns], []
                )]
            )

        return pd.concat(result_table)

    def test(self, date: datetime):
        return self._predict(date)


def test_model():
    tester = ModelTester(
        path_to_model=PATH_TO_MODEL,
        path_to_features=PATH_TO_FEATURES,
    )
    result = tester.test()
    result.to_csv(
        os.path.join(
            SAVE_PATH,
            f"result_{CRATE}.csv",
            sep=";",
            decimal=",",
        )
    )


def test_cut(dates: list):
    tester = CutTester(
        path_to_model=PATH_TO_MODEL,
        path_to_features=PATH_TO_FEATURES,
    )
    result = []
    for date in dates:
        result.append(tester.test(date))

    pd.concat(result).to_csv(
        os.path.join(
            SAVE_PATH,
            f"cut_result_{CRATE}.csv",
            sep=";",
            decimal=",",
        )
    )


if __name__ == '__main__':
    # dates = [
    #     datetime(year=2023, month=1, day=24, hour=0, minute=35, second=0),
    #     datetime(year=2023, month=1, day=24, hour=21, minute=50, second=0),
    # ]
    #
    # test_cut(dates)
    test_model()
