from typing import Dict

from components.segments_aggregator import (
    AggregatedSegment,
    AggregatedSourceDict,
)
from dto import SourceSettings


class Matcher:

    def __init__(self, sources_settings: Dict[str, SourceSettings]):
        self.sources_settings = sources_settings
        self.target_name = [
            name for name, src in self.sources_settings.items()
            if src.type == "target"
        ][0]

    def match_features_to_target(
        self, all_aggregated_sources: Dict[str, AggregatedSourceDict]
    ) -> (Dict[str, AggregatedSegment], Dict[str, Dict[str, float]]):
        targets_segments = {
            seg_key: all_aggregated_sources[self.target_name][seg_key]
            for seg_key in self.sources_settings[self.target_name
                                                 ].segments.keys()
        }
        for source_name, source_values in all_aggregated_sources.items():
            if source_values.is_target():
                continue
            for segment_name, segment_values in source_values.items():
                segment_settings = source_values.settings(
                ).segments[segment_name]
                for target_name in segment_settings.target_segments:
                    targets_segments[target_name].append_value(
                        segment_values.values()
                    )
        targets_dict = self._create_dict_from_segments(targets_segments)
        return targets_segments, targets_dict

    @staticmethod
    def _create_dict_from_segments(
        targets_segments: Dict[str, AggregatedSegment]
    ) -> Dict[str, Dict[str, float]]:
        targets_dict = {}
        for target_segment, aggregated_values in targets_segments.items():
            targets_dict[target_segment] = {
                str(value.transcription):
                float(value.value) if value.value else None
                for value in aggregated_values.values()
            }
        return targets_dict
