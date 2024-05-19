from copy import copy

import numpy as np
from source_data import Array


class FeaturesGenerator:

    def __init__(self):
        self.methods = {
            "abs": self.absolute,
            "norm": self.norm,
        }

    def get_method_values(self, method, *args, **kwargs):
        return self.methods[method](*args, **kwargs)

    @staticmethod
    def absolute(data_array: Array, *args, **kwargs) -> Array:
        """Модуль"""
        transcription = copy(data_array.transcription)
        transcription.add_tags({"preprocessing": ["abs"]}, )
        array = Array(
            transcription=transcription,
            values=abs(data_array.values),
            is_numeric=True,
            is_billet=False
        )
        return array

    @staticmethod
    def norm(data_array: Array, MIN, MAX, *args, **kwargs) -> Array:
        """Модуль"""
        transcription = copy(data_array.transcription)
        name = (
            data_array.transcription.workunit + "_"
            + data_array.transcription.name
        )
        transcription.add_tags({"preprocessing": ["norm"]})
        array = Array(
            transcription=transcription,
            values=np.array(
                [
                    (val - MIN[name]) / (MAX[name] - MIN[name])
                    for val in data_array.values
                ]
            ),
            is_numeric=True,
            is_billet=False
        )
        return array
