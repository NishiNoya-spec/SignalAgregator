from copy import copy
from typing import Dict, List

import constants
import numpy as np
import pandas as pd
from dto import SourceSettings
from naming import Transcription


class Array:

    def __init__(
        self,
        transcription: Transcription,
        values: np.array,
        is_numeric: bool,
        is_billet: bool,
    ):
        self.transcription = transcription
        self.is_numeric = is_numeric
        self.is_billet = is_billet
        self.values = values
        self.key = str(self.transcription)

    def __repr__(self):
        return (
            f"{str(self.transcription)}; "
            f"is_numeric={self.is_numeric}"
            f", is_billet={self.is_billet}"
        )

    def replace_values(self, new_values: np.array):
        self.values = new_values

    def append_keys_to_transcription(
        self, tags: Dict[str, List[str]], replace: bool = False
    ):
        new_transcription = self.transcription.add_tags(tags, replace)
        new_array = copy(self)
        new_array.transcription = new_transcription
        new_array.key = str(new_transcription)
        return new_array


class SourceDataset:

    def __init__(self, settings: SourceSettings):
        self.settings = settings
        self.data = {}
        self.billet_key = str(
            Transcription(
                workunit=self.settings.source,
                rolling_number=self.settings.rolling_number,
                name=self.settings.billet_column,
                interpolation="ni"
            )
        )

    def __iter__(self):
        self._data_keys = list(self.data.keys())
        self._index = -1
        return self

    def __next__(self):
        if len(self._data_keys) != len(list(self.data.keys())):
            raise RuntimeError("object changed size during iterations")
        if self._index < len(self.data) - 1:
            self._index += 1
            return self.data[self._data_keys[self._index]]
        else:
            raise StopIteration

    def __getitem__(self, key: str):
        return self.data[key]

    def __repr__(self):
        return f"{self.settings.source}; " \
               f"billet='{self.settings.billet_column}'"

    def billet_array(self):
        """Возвращает Array, для которого is_billet = True"""
        return self.data[self.billet_key]

    def append_dataframe_to_source_data(self, data: pd.DataFrame):
        """Формирует из DataFrame набор данных в формате Dict[Array]"""
        for name, values in data.iteritems():
            values = values.to_numpy()
            is_numeric = self._is_numeric(values)
            transcription = Transcription(
                workunit=self.settings.source,
                rolling_number=self.settings.rolling_number,
                name=str(name),
                interpolation="ni"
            )
            is_billet = (
                True if str(transcription) == self.billet_key else False
            )
            array = Array(
                transcription=transcription,
                values=values,
                is_numeric=is_numeric,
                is_billet=is_billet
            )
            self.data[str(transcription)] = array

    def append_array_to_source_data(self, array: Array, old_key: str = None):
        """Добавляет новый Array"""
        self.data[array.key] = array
        if old_key == self.billet_key:
            self.billet_key = str(array.transcription)

    def replace_array_values(self, key: str, new_values: np.array):
        """Изменяет значения определнного Array"""
        self.data[key].replace_values(new_values)

    def remove_arrays(self, arrays: List[str]):
        for key in arrays:
            self.data.pop(key)

    def append_tags_to_array(
        self, key: str, tags: Dict[str, List[str]], replace: bool = False
    ):
        new_array = self.data[key].append_keys_to_transcription(tags, replace)
        self.remove_arrays([key])
        self.append_array_to_source_data(new_array, key)

    def append_tags_to_all_arrays(
        self, tags: Dict[str, List[str]], replace: bool = False
    ):
        init_keys = self.data.keys()
        for key in init_keys:
            new_array = self.data[key].append_keys_to_transcription(
                tags, replace
            )
            self.remove_arrays([key])
            self.append_array_to_source_data(new_array, key)

    def return_arrays_by_tags(self, tags: Dict[str, str]) -> List[Array]:
        arrays = []
        for array in self:
            for tag_key, tag_value in tags.items():
                if getattr(array.transcription, tag_key) != tag_value:
                    break
                arrays.append(array)
        return arrays

    @staticmethod
    def _is_numeric(values: np.array) -> bool:
        column_type = values.dtype
        is_numeric = False
        for parent_type in constants.AVAILABLE_TYPES:
            if np.issubdtype(column_type, parent_type):
                is_numeric = True
                break
        return is_numeric
