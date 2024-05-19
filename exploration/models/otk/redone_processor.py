from glob import glob

import pandas as pd
from tqdm import tqdm

from exploration.models.otk import constants


class RedoneGetter:

    def __init__(self, data_path):
        self.data = self._get_redone_data(data_path)

    def _get_redone_data(self, data_path):
        general_redone = list()
        for file_name in tqdm(glob(data_path)):
            df = pd.read_excel(
                file_name,
                usecols=[
                    constants.BILLET_ID,
                    "Рельс",
                    constants.MOMENT,
                    "Дефект",
                ],
            )

            df[constants.DEFECT_CRATE] = df["Рельс"].str.replace(r'\d+', '')
            df = df[df[constants.DEFECT_CRATE].str.contains('[ABCY]')]
            df[constants.BILLET_ID] = df[constants.BILLET_ID].str[:9]
            defect_type_filter = df["Дефект"].str.contains(
                '|'.join(constants.DEFECT_TYPE)
            ) | (df["Дефект"] == "")
            df = df[defect_type_filter]
            df[constants.DEFECT_COL] = (df["Дефект"] != "").astype(int)

            df[constants.BILLET_ID] = (
                df[constants.BILLET_ID] + "_"
                + pd.to_datetime(df[constants.MOMENT]).dt.year.astype(str)
            )
            general_redone.append(
                df[[
                    constants.MOMENT,
                    constants.BILLET_ID,
                    constants.DEFECT_COL,
                    constants.DEFECT_CRATE,
                ]]
            )
        return pd.concat(general_redone)
