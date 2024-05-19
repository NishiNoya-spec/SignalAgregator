from copy import copy
from typing import Dict

import pandas as pd
from dto import PipelineSetup, SourceSettings
from handlers import factory
from matcher import Matcher


class Processer:

    def __init__(
        self,
        handlers_factory: factory.HandlersFactory,
        metadata: pd.DataFrame,
        setup: PipelineSetup,
        data_mapping: dict,
        sources_settings: Dict[str, SourceSettings],
        matcher: Matcher,
    ):
        self.handlers_factory = handlers_factory
        self.matcher = matcher
        self.metadata = metadata
        self.constants = setup
        self.data_mapping = data_mapping
        self.sources_settings = sources_settings

    def create_data_batch(self, billet_id: str):
        all_aggregated_sources = {}
        for source, filepath in self.data_mapping[billet_id].items():
            settings = copy(self.sources_settings[source])
            data = pd.read_csv(
                filepath, sep=";", decimal=",", encoding=settings.encoding
            )
            source_handler = (
                self.handlers_factory.get_handler(settings.handler)
            )
            all_aggregated_sources[source] = source_handler.process_pipeline(
                billet_id, data, settings
            )
        targets_segments, targets_dict = self.matcher.match_features_to_target(
            all_aggregated_sources
        )
        return targets_segments, targets_dict
