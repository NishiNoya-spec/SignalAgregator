import os

import pandas as pd

from exploration.models.otk import constants
from exploration.models.otk.cut_processor import CutProcessor
from exploration.models.otk.its_processor import ITSGetter
from exploration.models.otk.redone_processor import RedoneGetter
from exploration.models.otk.torsion_getter import TorsionGetter

PATH_TO_ITS_DATA = r"\\ZSMK-9684-001\Data\DS\ИТС + Передел\ITS"
PATH_TO_REDONE_DATA = r"\\ZSMK-9684-001\Data\DS\ИТС + Передел\redone"
SAVE_PATH = r"\\ZSMK-9684-001\Data\DS\ИТС + Передел\results"
PATH_TO_METADATA = r"\\ZSMK-9684-001\Data\DS\metadata"
PATH_TO_IPSRNKA_DATA = "path/to/ipsrnk_data"


def get_result_file_name(moments):
    moments = pd.to_datetime(moments)
    min_time = moments.dt.date.min().strftime("%Y_%m_%d")
    max_time = moments.dt.date.max().strftime("%Y_%m_%d")
    return f"{min_time}_to_{max_time}.csv"


def save_result(result_df):
    save_file_name = get_result_file_name(result_df[constants.MOMENT])
    result_df.to_csv(os.path.join(SAVE_PATH, save_file_name))


if __name__ == '__main__':
    print("getting its data ...")
    its_getter = ITSGetter(os.path.join(PATH_TO_ITS_DATA, "*.xlsx"))

    print("getting redone data ...")
    redone_getter = RedoneGetter(os.path.join(PATH_TO_REDONE_DATA, "*.xlsx"))

    print("getting cut data ...")
    cut_processor = CutProcessor(os.path.join(PATH_TO_METADATA, "*.xlsx"))

    print("getting torsion data ...")
    torsion_getter = TorsionGetter(os.path.join(PATH_TO_IPSRNKA_DATA, "*.csv"))

    result_df = [
        its_getter.data,
        redone_getter.data,
    ]

    result_df = pd.concat(result_df)
    result_df = result_df.groupby(
        [
            constants.BILLET_ID,
            constants.DEFECT_CRATE,
        ]
    ).max().reset_index().drop_duplicates(subset=[constants.BILLET_ID])
    result_df = result_df.merge(cut_processor.data, on=constants.BILLET_ID)
    result_df = result_df.merge(
        torsion_getter.data.drop_duplicates(subset=[constants.BILLET_ID]),
        on=constants.BILLET_ID, )
    save_result(result_df.drop_duplicates(subset=[constants.BILLET_ID]))
