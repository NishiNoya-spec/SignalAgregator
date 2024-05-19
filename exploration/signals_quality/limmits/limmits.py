import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from exploration.models.otk import constants
from exploration.utils.metadata import MethaDataProcessor

USE_COLUMNS = [
    "billet_id",
    # Название сигналов для подсчета лимитов
]
PATH_TO_PREPROCESSED_DATA = r'\\ZSMK-9684-001\Data\DS' \
                            r'\u0_limits\2023-03-27_2023-11-17\*.csv'

SAVE_PATH = "stats/model_stats/"

PATH_TO_METHADATA = r'\\ZSMK-9684-001\Data\DS\metadata\*xlsx'

if __name__ == "__main__":
    paths = glob(PATH_TO_PREPROCESSED_DATA)
    data_norm = list()
    for path in tqdm(paths[:]):
        data = pd.read_csv(path, usecols=USE_COLUMNS)
        use_columns = [
            col for col in data.columns
            if "norm" not in col and "pos_up_hor_roll" in col in [
                "billet_id",
                "pos_vert_roll",
                "pos_low_hor_roll",
                "load_hor_roll",
                "gr24_bt01.temp.extrapolated",
            ]
        ]
        data_norm.append(
            data[use_columns].rename(
                columns={
                    col: col.replace("[0_end]", "[0_103]")
                    for col in use_columns
                }
            )
        )

    data_norm = pd.concat(data_norm).set_index("billet_id")
    metadata_processor = MethaDataProcessor(PATH_TO_METHADATA)
    moments = metadata_processor.get_billets_rolling_time()

    data_norm = pd.merge(
        moments, data_norm, left_index=True, right_index=True
    ).set_index(constants.MOMENT).sort_index()

    stats = dict()
    for col in tqdm(data_norm.columns):
        print()
        if col != "billet_id":
            chack_data = data_norm[col].dropna()
            if len(chack_data) == 0:
                print(col)
                continue

            plt.plot(chack_data.index, chack_data)

            min_val = chack_data.quantile(0.015)
            max_val = chack_data.quantile(0.995)

            stats[col] = [min_val, max_val]
            plt.axhline(y=max_val, color='r', linestyle='-')
            plt.axhline(y=min_val, color='r', linestyle='-')

            plt.savefig(os.path.join(SAVE_PATH, f'{col}_graph.png'))
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()

            chack_data.plot.box(title=col, xticks=[])
            plt.savefig(os.path.join(SAVE_PATH, '{col}_boxplot.png'))
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()

            plt.hist(chack_data)
            plt.savefig(os.path.join(SAVE_PATH, '{col}_req.png'))
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()
            chack_data.to_csv(
                os.path.join(
                    SAVE_PATH,
                    "{col}_seq.csv",
                    sep=";",
                    decimal=",",
                )
            )
            data_norm.replace(
                [np.inf, -np.inf], chack_data.quantile(0.995), inplace=True
            )
    pd.DataFrame(stats).T.to_csv(
        os.path.join(
            SAVE_PATH,
            "model__limits.csv",
            sep=";",
            decimal=",",
        )
    )
