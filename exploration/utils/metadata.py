from glob import glob

import pandas as pd

from exploration.models.otk import constants


class MethaDataProcessor:

    def __init__(self, path_to_metadata: str):
        self.data = self._get_metadata(path_to_metadata)
        self._set_billet_id()

    def _get_metadata(self, path_to_metadata):
        billets_metadata = list()
        for metadata_path in glob(path_to_metadata):
            metadata = pd.read_excel(
                metadata_path,
                engine='openpyxl',
                usecols=[
                    "Плавка",
                    "Руч",
                    "Заг",
                    "Вр.реза",
                    "Дл. ЛНК, мм",
                    "Дл. СОС, мм",
                    "Марка",
                ],
            )
            billets_metadata = pd.concat(metadata)

        return (
            billets_metadata.drop_duplicates().set_index("billet_id").rename(
                columns={
                    "Вр.реза": constants.MOMENT,
                    "Марка": constants.STEEL_MARK,
                }
            )
        )

    def _set_billet_id(self, ):
        self.data = (
            "Л2" + self.data["Плавка"].str.split("-", expand=True).iloc[:, -1]
            + self.data["Руч"].astype(str) + "0" + self.data["Заг"].astype(str)
            + "_" + self.data[constants.MOMENT].dt.year.astype(str)
        )

    def get_billets_rolling_time(self):
        return self.data[[constants.MOMENT]].dropna()
