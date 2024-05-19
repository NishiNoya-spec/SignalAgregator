from exploration.models.otk import constants
from exploration.utils.metadata import MethaDataProcessor


class CutProcessor(MethaDataProcessor):

    def __init__(self, path_to_metadata):
        super().__init__(path_to_metadata)
        self._set_cut_columns()

    def _set_cut_columns(self, ):
        self.data["dl"] = self.data["Дл. ЛНК, мм"] - self.data["Дл. СОС, мм"]
        self.data["dlA"] = self.data["dl"] * (
            (700 < self.data["dl"]) & (self.data["dl"] < 2000)
        ).astype(int)
        self.data["dlY"] = (
            self.data["Дл. ЛНК, мм"] - self.data["dlA"] - 100000
        ) * (self.data["dlA"] != 0).astype(int)
        self.data = self.data[[
            constants.BILLET_ID, "dlA", "dlY", constants.STEEL_MARK
        ]].drop_duplicates(subset=[constants.BILLET_ID])
