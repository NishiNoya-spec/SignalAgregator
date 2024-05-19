from dataclasses import dataclass
from typing import Dict

import aggregator
import dto
import numpy as np
import pandas as pd
from dto import Source


@dataclass
class Decomposer:
    constants: dto.Constants
    aggregator = aggregator.AggregationFactory()
    sources_settings: Dict[str, Source]

    def aggregate_segments(self, data: pd.DataFrame, source: str) -> dict:
        aggregated_segments = {}
        settings = self.sources_settings[source]
        for seg_id, segment in settings.segments.items():

            # Начальная точка
            if segment.start_point < 0:
                start_point = max(data[settings.billet_column]) + \
                              segment.start_point
            else:
                start_point = segment.start_point

            # Конечная точка
            if segment.end_point == "end":
                end_point = max(data[settings.billet_column])
            elif segment.end_point < 0:
                end_point = max(data[settings.billet_column]) + \
                            segment.end_point
            else:
                end_point = segment.end_point

            # Сегментация
            segment_data = data.loc[
                (data[settings.billet_column] >= start_point)
                & (data[settings.billet_column] <= end_point)]
            aggregated_segments[seg_id] = self._agg_by_cols(
                segment_data, settings,
                f"{segment.start_point}_{segment.end_point}"
            )
        return aggregated_segments

    def aggregate_by_method(self, data: dict, billet_id: str):
        target_sources = [
            source for source, settings in self.sources_settings.items()
            if settings.type == "target"
        ]
        aggregated_dict = {}
        for tg_source in target_sources:
            for tg_segment, tg_values in data[tg_source].items():
                aggregated_dict[tg_segment] = tg_values
                aggregated_dict[tg_segment]["billet_id"] = billet_id
                for ft_source, ft_segments in data.items():
                    for ft_segment, ft_values in ft_segments.items():
                        ft_set = self.sources_settings[ft_source]
                        seg_set = ft_set.segments[ft_segment]
                        if (ft_set.type != "feature"
                                or tg_segment not in seg_set.target_segments):
                            continue
                        aggregated_dict[tg_segment].update(ft_values)
        return list(aggregated_dict.values())

    def _agg_by_cols(
        self, segment_data: np.array, settings: Source, key: str
    ) -> dict:
        segment_array_billet_points = segment_data[settings.billet_column
                                                   ].to_numpy()
        segment_data = segment_data.drop([settings.billet_column], axis=1)
        segment_array = segment_data.to_numpy()
        aggregated_segment = {}
        for idx, colname in zip(range(segment_array.shape[1]),
                                segment_data.columns):
            for method in settings.aggregation_methods:
                aggregated_segment[f'{colname}_[{key}]_[{method}]'] = \
                    self.aggregator.agg_by_method(
                        segment_array[:, idx],
                        method,
                        points=segment_array_billet_points)
        return aggregated_segment
