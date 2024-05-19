# -*- coding: utf-8 -*-
import os
from glob import glob

import pandas as pd

from exploration.prototypes.converter.constants import (
    CONVERTOR_INPUT_DATA,
    CONVERTOR_OUTPUT_DIR,
)
from exploration.prototypes.converter.core.converter import PATH_TO_DATA
from exploration.prototypes.converter.utils.converter_utils import convert

if __name__ == "__main__":
    calculation_types = {
        "Конвертор": [
            os.path.join(PATH_TO_DATA, CONVERTOR_INPUT_DATA),
            os.path.join(PATH_TO_DATA, CONVERTOR_OUTPUT_DIR)
        ],
    }
    while True:
        current_status = []
        download_buttons = []
        for calculation_type, dirs in calculation_types.items():
            if os.path.exists(os.path.join(dirs[1], "alert.txt")):
                pass
            else:
                if os.path.exists(os.path.join(dirs[1], "settings.csv")):
                    if CONVERTOR_OUTPUT_DIR in dirs[1]:
                        try:
                            convertor_settings = pd.read_csv(
                                os.path.join(dirs[1], "settings.csv")
                            )

                            if "data_path" in convertor_settings.columns:
                                data_path = convertor_settings["data_path"][0]
                            else:
                                data_path = dirs[0]

                            convert(
                                [
                                    [] if flag == '[]' else [''] for flag in
                                    convertor_settings["filter_flag"].to_list()
                                ], [
                                    [] if flag == '[]' else [''] for flag in
                                    convertor_settings["abs_flag"].to_list()
                                ], glob(os.path.join(data_path, "*.csv")), [
                                    signal.split(",") for signal in
                                    convertor_settings["signals_dropdown"].
                                    to_list()
                                ], convertor_settings["methods_dropdown"].
                                to_list(),
                                convertor_settings["zones_dropdown"].to_list(),
                                convertor_settings["zones_min_len"].to_list(),
                                convertor_settings["zones_max_len"].to_list(),
                                dirs[1]
                            )
                            os.remove(os.path.join(dirs[1], "settings.csv"))
                        except pd.errors.EmptyDataError:
                            pass
                        except FileNotFoundError:
                            print("File not found")
                        except:    # noqa
                            print("Data format ERROR")
                            pd.DataFrame().to_csv(
                                os.path.join(dirs[1], "alert.txt")
                            )
                            break
