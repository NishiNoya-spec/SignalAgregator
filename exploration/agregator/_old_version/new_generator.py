from dataclasses import dataclass

from constants import MIN, DIFF_MAX
from dto import Source

import pandas as pd

import dto


@dataclass
class DataGenerator:
    constants: dto.Constants

    @staticmethod
    def generate_secondary_features(data: pd.DataFrame, settings: Source):
        for colname_1, values_1 in data.iteritems():
            # Пропускаем колонку с биллетами
            if colname_1 == settings.billet_column:
                continue

            # Синтетические фичи
            if "abs" in settings.synthetic_methods:
                try:
                    data[f"abs_{colname_1}"] = abs(values_1)
                except:
                    data[f"abs_{colname_1}"] = values_1
            if "norm" in settings.synthetic_methods:
                try:
                    data[f"norm_{colname_1}"] = (values_1 - MIN[colname_1]) / (DIFF_MAX[colname_1] - MIN[colname_1])
                except KeyError:
                    pass
            if "mult" in settings.synthetic_methods:
                for colname_2, values_2 in data.iteritems():
                    if colname_1 == settings.billet_column:
                        continue
                    data[f"mult_{colname_1}_{colname_2}"] = values_1 * values_2
        return data
