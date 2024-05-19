from abc import ABC, abstractmethod

import pandas as pd
from components.features_generator import FeaturesGenerator
from components.filters import Filters
from components.secondary_functions import SecondaryFunctions
from components.segments_aggregator import SegmentsAggregator
from dto import SourceSettings


class SourceHandler(ABC):
    source: str

    @abstractmethod
    def __init__(
        self, features_generator: FeaturesGenerator, filters: Filters,
        secondary_functions: SecondaryFunctions,
        segments_aggregator: SegmentsAggregator
    ):
        self.segments_aggregator = segments_aggregator
        self.secondary_functions = secondary_functions
        self.feature_generator = features_generator
        self.filters = filters

    @abstractmethod
    def process_pipeline(
        self, billet_id: str, data: pd.DataFrame, settings: SourceSettings
    ):
        ...
