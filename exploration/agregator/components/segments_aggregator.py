from typing import Dict, List, Optional

import numpy as np
from dto import Segment, SourceSettings
from naming import Transcription
from source_data import Array


class AggregatedValue:

    def __init__(
        self,
        segment_id: str,
        transcription: Transcription,
        value: Optional[float],
        is_bad: bool = False,
        bad_reason: str = None
    ):
        self.segment_id = segment_id
        self.transcription = transcription
        self.value = value
        self.is_bad = is_bad
        self.bad_reason = bad_reason

    def __repr__(self):
        return f"{str(self.transcription)}, value={self.value}"


class AggregatedSegment(dict):

    def __init__(
        self,
        name: str,
        start_point: float,
        end_point: float,
        values: Dict[str, AggregatedValue] = ()
    ):
        self._name = name
        self._start_point = start_point
        self._end_point = end_point
        super(AggregatedSegment, self).__init__(values)

    def append_value(self, values: List[AggregatedValue]):
        for value in values:
            self[str(value.transcription)] = value

    def name(self):
        return self._name

    def start_point(self):
        return self._start_point

    def end_point(self):
        return self._end_point


class AggregatedSourceDict(dict):

    def __init__(
        self,
        settings: SourceSettings,
        segments: Dict[str, AggregatedSegment] = ()
    ):
        super(AggregatedSourceDict, self).__init__(segments)
        self._source = settings.source
        self._is_target = True if settings.type == 'target' else False
        self._settings = settings

    def __repr__(self):
        count_bad = sum(
            [
                1 for seg_vals in self.values() for value in seg_vals.values()
                if value.is_bad
            ]
        )
        count_values = sum(len(seg_vals) for seg_vals in self.values())
        return f"AggregatedSource(total_values={count_values}, " \
               f"bad_values={count_bad})"

    def append_segment(self, segment: AggregatedSegment):
        self[segment.name()] = segment

    def source(self):
        return self._source

    def settings(self):
        return self._settings

    def is_target(self):
        return self._is_target


class SegmentsAggregator:

    def __init__(self):
        self.methods = {
            "median": self.median_aggregate,
            "min": self.min_aggregate,
            "max": self.max_aggregate,
            "mean": self.mean_aggregate,
            "tg": self.tg_aggregate,
        }

    @staticmethod
    def return_segment_values(
        segment: Segment, billet_array: Array, data_array: Array
    ) -> np.array:
        # Начальная точка
        if segment.start_point < 0:
            start_point = max(billet_array.values) + segment.start_point
        else:
            start_point = segment.start_point

        # Конечная точка
        if segment.end_point == "end":
            end_point = max(billet_array.values)
        elif segment.end_point < 0:
            end_point = max(billet_array.values) + segment.end_point
        else:
            end_point = segment.end_point

        # Сегментация
        points = billet_array.values[(billet_array.values < 3)
                                     & (billet_array.values > 0)]
        segment_data = [
            value for bil, value in zip(billet_array.values, data_array.values)
            if end_point >= bil >= start_point
        ]
        return np.array(points), np.array(segment_data)

    def get_method_value(
        self,
        segment_id: str,
        transcription: Transcription,
        method: str,
        *args,
        **kwargs,
    ) -> AggregatedValue:
        return AggregatedValue(
            segment_id, transcription, self.methods[method](*args, **kwargs)
        )

    @staticmethod
    def median_aggregate(data_array: np.array):
        return np.median(data_array)

    @staticmethod
    def max_aggregate(data_array: np.array):
        return np.max(data_array)

    @staticmethod
    def min_aggregate(data_array: np.array):
        return np.min(data_array)

    @staticmethod
    def mean_aggregate(data_array: np.array):
        return np.mean(data_array)

    @staticmethod
    def tg_aggregate(data_array: np.array, points: np.array):
        extended_points = np.c_[points, np.ones(len(points))]
        tg, _ = np.linalg.lstsq(extended_points, data_array, rcond=None)[0]

        return tg

    def _rotate(self, data_array: np.array, points: np.array):
        angle = np.arctan(self.tg_aggregate(data_array, points))

        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle),
                                               np.cos(angle)]]
        )

        # Смещаем график к началу координат
        x_centered = points - np.mean(points)
        y_centered = data_array - np.mean(data_array)

        # Поворачиваем график
        xy_rotated = np.dot(
            rotation_matrix, np.vstack([x_centered, y_centered])
        )

        # Возвращаем график в исходное положение
        rotated_points = xy_rotated[0] + np.mean(points)
        rotated_array = xy_rotated[1] + np.mean(data_array)

        return rotated_points, rotated_array
