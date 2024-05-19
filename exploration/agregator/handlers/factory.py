from components.features_generator import FeaturesGenerator
from components.filters import Filters
from components.secondary_functions import SecondaryFunctions
from components.segments_aggregator import SegmentsAggregator
from dto import Materials
from handlers.BASE_handler import BASEHandler
from handlers.BASE_WITH_POINTS_handler import BASEWITHPOINTSHandler
from handlers.interfaces import SourceHandler
from handlers.WBF_PIRO_handler import WBFPIROHandler
from handlers.WBF_SINGLE_handler import WBFSINGLEHandler


class HandlersFactory:

    def __init__(
        self, features_generator: FeaturesGenerator, filters: Filters,
        materials: Materials, secondary_functions: SecondaryFunctions,
        segments_aggregator: SegmentsAggregator
    ):
        handler_setup = {
            "features_generator": features_generator,
            "filters": filters,
            "secondary_functions": secondary_functions,
            "segments_aggregator": segments_aggregator,
            "materials": materials
        }
        self.handlers_dict = {
            BASEHandler.source: BASEHandler(**handler_setup),
            BASEWITHPOINTSHandler.source: BASEHandler(**handler_setup),
            WBFPIROHandler.source: WBFPIROHandler(**handler_setup),
            WBFSINGLEHandler.source: WBFSINGLEHandler(**handler_setup)
        }

    def get_handler(self, source: str) -> SourceHandler:
        return self.handlers_dict[source]
