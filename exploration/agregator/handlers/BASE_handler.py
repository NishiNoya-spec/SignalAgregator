import json

import pandas as pd
from components.features_generator import FeaturesGenerator
from components.filters import Filters
from components.secondary_functions import SecondaryFunctions
from components.segments_aggregator import (
    AggregatedSegment,
    AggregatedSourceDict,
    AggregatedValue,
    SegmentsAggregator,
)
from dto import Materials, SourceSettings
from handlers.interfaces import SourceHandler
from source_data import SourceDataset


class BASEHandler(SourceHandler):
    source = "BASE"

    def __init__(
        self,
        features_generator: FeaturesGenerator,
        filters: Filters,
        materials: Materials,
        secondary_functions: SecondaryFunctions,
        segments_aggregator: SegmentsAggregator,
    ):
        self.materials = materials
        self.segments_aggregator = segments_aggregator
        self.secondary_functions = secondary_functions
        self.feature_generator = features_generator
        self.filters = filters

    def process_pipeline(
        self, billet_id: str, data: pd.DataFrame, settings: SourceSettings
    ) -> AggregatedSourceDict:
        source_data = SourceDataset(settings)

        # Подготовительный пайплайн
        source_data.append_dataframe_to_source_data(data)
        source_data = (
            self.secondary_functions.
            sort_dataset_ascending_by_billet(source_data)
        )
        source_data = self.secondary_functions.approximate_billet_by_bounds(
            source_data
        )
        with open(self.materials.PATHS['workcenters']) as handle:
            workcenters = json.load(handle)
        source_data = (
            self.secondary_functions.append_workcenter_to_transcription(
                source_data, workcenters
            )
        )

        # Базовый пайплайн
        source_data = self.filter_data(source_data)
        source_data = self.generate_features(source_data)
        aggregated_source = self.calculate_aggregations(source_data)
        return aggregated_source

    def filter_data(self, source_data: SourceDataset) -> SourceDataset:
        """Фильтрация данных"""
        filtered_columns = []
        for data_array in source_data:
            if data_array.is_billet:
                continue
            for method in [*source_data.settings.filtration_methods,
                           "forbidden_columns"]:
                is_bad = self.filters.filter_by(
                    method=method,
                    data_array=data_array,
                    source_data=source_data
                )
                if is_bad:
                    filtered_columns.append(data_array.transcription)
        source_data.remove_arrays(filtered_columns)
        return source_data

    def generate_features(self, source_data: SourceDataset) -> SourceDataset:
        """Генерация вторичных фичей"""
        new_arrays = []
        for method in source_data.settings.secondary_features:
            for data_array in source_data:
                if data_array.is_billet or not data_array.is_numeric:
                    continue
                new_arrays.append(
                    self.feature_generator.get_method_values(
                        method, data_array
                    )
                )
        for new_array in new_arrays:
            source_data.append_array_to_source_data(new_array)
        return source_data

    def calculate_aggregations(
        self, source_data: SourceDataset
    ) -> AggregatedSourceDict:
        """Аггрегация данных по сегментам"""
        aggregated_source = AggregatedSourceDict(source_data.settings)
        for segment_id, segment in source_data.settings.segments.items():
            aggregated_segment = AggregatedSegment(
                segment_id, segment.start_point, segment.end_point
            )
            for data_array in source_data:
                if data_array.is_billet:
                    continue
                for method in source_data.settings.aggregation_methods:
                    transcription = data_array.transcription.add_tags(
                        {
                            "sector_range": [f"{segment.start_point}_{segment.end_point}"],
                            "aggregation": [f"{method}"]
                        }
                    )
                    segment_values = (
                        self.segments_aggregator.return_segment_values(
                            segment, source_data.billet_array(), data_array
                        )
                    )
                    segment_values = segment_values[~pd.isnull(segment_values)]
                    if not data_array.is_numeric:
                        aggregated_value = AggregatedValue(
                            segment_id,
                            transcription,
                            None,
                            True,
                            "Not numeric",
                        )
                    elif len(segment_values) == 0:
                        aggregated_value = AggregatedValue(
                            segment_id, transcription, None, True, "Empty"
                        )
                    else:
                        aggregated_value = (
                            self.segments_aggregator.get_method_value(
                                segment_id, transcription, method,
                                segment_values
                            )
                        )
                    aggregated_segment.append_value([aggregated_value])
            aggregated_source.append_segment(aggregated_segment)
        return aggregated_source
