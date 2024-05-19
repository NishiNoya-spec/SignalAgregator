from datetime import datetime
from glob import glob

import pandas as pd
from tqdm import tqdm

from exploration.models.otk import constants


class ITSProcessor:

    def __init__(self, path_to_its_file):
        self.data = pd.read_excel(
            path_to_its_file, usecols=constants.RAIL_INFO + constants.DEFECTS
        ).fillna("")
        self._process()

    def _process(self, ):
        self.data = self.data.rename(
            columns={
                "загот": constants.BILLET_ID,
                "дата_прокатки": constants.MOMENT
            }
        )
        self._filter_by_mark()
        self._set_defect()
        self._add_year_to_id()

        self.data = self.data.drop(
            [
                constants.STEEL_MARK,
                "рельс",
            ] + constants.DEFECTS,
            axis=1,
        )

    def _filter_by_mark(self):
        self.data = self.data[self.data[constants.STEEL_MARK].isin["Э76ХФ",
                                                                   "Э90ХАФ",
                                                                   "76ХФ",
                                                                   "90ХАФ",
                                                                   ]]

    def _set_defect(self):
        self.data[constants.DEFECT_COL
                  ] = self.data[[defect for defect in constants.DEFECTS
                                 ]].sum(axis=1)
        defect_type_filter = self.data[constants.DEFECT_COL].str.contains(
            '|'.join(constants.DEFECT_TYPE)
        ) | (self.data[constants.DEFECT_COL] == "")
        self.data = self.data[defect_type_filter]
        self.data[constants.DEFECT_COL
                  ] = (self.data[constants.DEFECT_COL] != "").astype(int)
        self.data[constants.DEFECT_CRATE
                  ] = self.data["рельс"].str.replace(r'\d+', '')

    def _add_year_to_id(self):
        self.data[constants.BILLET_ID] = (
            self.data[constants.BILLET_ID].str[:-1] + "_"
            + self.data[constants.MOMENT].dt.year.astype(str)
        )


class ITSGetter:

    def __init__(self, path_to_its_data):
        self.path_to_its_data = path_to_its_data
        self.data = self._get_its_data()

    def _get_its_data(self):
        its_data = []
        for file_path in tqdm(glob(self.path_to_its_data)):
            its_processor = ITSProcessor(file_path)
            its_data.append(its_processor.data)

        return pd.concat(its_data)

    def get_from_period(self, start_date: datetime, end_date: datetime):
        return self.data[
            (self.data[constants.MOMENT].dt.date >= start_date.date())
            & (self.data[constants.MOMENT].dt.date <= end_date.date())]
