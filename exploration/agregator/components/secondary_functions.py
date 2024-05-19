from typing import Dict, List

import constants
import numpy as np
import pandas as pd
from dto import CutPoint, Segment
from source_data import SourceDataset


class SecondaryFunctions:

    def __init__(self):
        pass

    @staticmethod
    def sort_dataset_ascending_by_billet(
        source_data: SourceDataset
    ) -> SourceDataset:
        """Сортировка billet points по возрастанию"""
        new_sequence = sorted(
            enumerate(source_data.billet_array().values), key=lambda x: x[1]
        )
        new_sequence = [val[0] for val in new_sequence]
        for data_array in source_data:
            new_values = [data_array.values[index] for index in new_sequence]
            source_data.append_tags_to_array(
                data_array.key, {"secondary_functions": ["sortas"]}
            )
            source_data.replace_array_values(
                data_array.key, np.array(new_values)
            )
        return source_data

    @staticmethod
    def approximate_billet_by_bounds(
        source_data: SourceDataset
    ) -> SourceDataset:
        """Аппроксимация столбца Billet к
        необходимой, строго фиксированной размерности"""
        ub = source_data.settings.interpolation
        lb = 0
        max_billet = max(source_data.billet_array().values)
        new_billet_values = (
            source_data.billet_array().values * (ub - lb) / max_billet
        ) + lb
        source_data.replace_array_values(
            source_data.billet_key, new_billet_values
        )
        source_data.append_tags_to_all_arrays(
            {"interpolation": [f"[i_{lb:.1f}_{ub:.1f}]"]}, replace=True
        )
        return source_data

    @staticmethod
    def append_workcenter_to_transcription(
        source_data: SourceDataset, workcenters: Dict[str, str]
    ) -> SourceDataset:
        for data_array in source_data:
            tags = workcenters[data_array.transcription.workunit]
            source_data.append_tags_to_array(
                data_array.key, {"workcenter": [tags]}, replace=True
            )
        return source_data

    @staticmethod
    def cut_wbf_pirometer_signals(
        source_data: SourceDataset, piro_settings: Dict[str, Dict[str, str]]
    ):
        all_target_segments = list(
            set(
                [
                    target for seg in source_data.settings.segments.values()
                    for target in seg.target_segments
                ]
            )
        )
        all_cut_indexes = []
        for name, points in piro_settings["SETTINGS"].items():
            array = source_data.return_arrays_by_tags({"name": name})[0]

            # Ищем точки разделения
            cut_indexes = []
            point_idx = 0
            col_len = len(array.values)
            for idx in range(col_len):
                point_data = CutPoint(*points[point_idx])
                if (idx - point_data.L_WIN < 0
                        or idx + point_data.R_WIN >= col_len):
                    continue
                l_delta = (
                    array.values[idx] - array.values[idx - point_data.L_WIN]
                )
                r_delta = (
                    array.values[idx + point_data.R_WIN] - array.values[idx]
                )
                if (point_data.L_DELTA_MIN <= l_delta <= point_data.L_DELTA_MAX
                        and (point_data.R_DELTA_MIN <= r_delta <=
                             point_data.R_DELTA_MAX)):
                    cut_indexes.append(idx)
                    if point_idx + 1 < len(points):
                        point_idx += 1
                    else:
                        break
            all_cut_indexes.extend(cut_indexes)
            for number, indexes in enumerate(zip(cut_indexes[:-1],
                                                 cut_indexes[1:])):
                source_data.settings.segments[f"{name}_{number}"] = Segment(
                    start_point=source_data.billet_array().values[indexes[0]],
                    end_point=source_data.billet_array().values[indexes[1]],
                    target_segments=all_target_segments
                )
        all_cut_indexes = sorted(all_cut_indexes)
        for number, indexes in enumerate(zip(all_cut_indexes[:-1],
                                             all_cut_indexes[1:])):
            source_data.settings.segments[f"ALL_PTS_{number}"] = Segment(
                start_point=source_data.billet_array().values[indexes[0]],
                end_point=source_data.billet_array().values[indexes[1]],
                target_segments=all_target_segments
            )
        return source_data

    @staticmethod
    def convert_date_columns_to_numeric(
        source_data: SourceDataset, tags_list: List[Dict[str, str]]
    ) -> SourceDataset:
        for tag in tags_list:
            tag_arrays = source_data.return_arrays_by_tags(tag)
            for array in tag_arrays:
                date_values = [
                    (pd.Timestamp(value) - constants.BASE_TIME).total_seconds()
                    for value in array.values
                ]
                min_data = min(date_values)
                converted_values = [value - min_data for value in date_values]
                source_data.replace_array_values(
                    array.key, np.array(converted_values)
                )
                source_data.append_tags_to_array(
                    array.key, {"secondary_functions": ["converted"]}
                )
        return source_data
