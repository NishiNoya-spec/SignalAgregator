import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import pickle
import math
from multiprocessing import cpu_count, pool
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import catboost

ABSPATH = r"Z:\DS\WBF_tail_full\2023-03-27_2023-09-30"
FORBIDDEN_COLUMNS = []
NUM_OF_CORES = 18
path_to_files = os.path.join(ABSPATH, "*csv")
path_to_result = os.path.join(ABSPATH, "concated")


def concat_files():
    print(f"Конкатенация файлов в общую таблицу")
    base_df = pd.DataFrame()
    filepaths = glob(path_to_files)
    filepath_batches = []
    for n in range(math.ceil(len(filepaths) / NUM_OF_CORES)):
        filepath_batches.append(
            filepaths[(NUM_OF_CORES * n):(NUM_OF_CORES * (n + 1))])

    for num, batch in tqdm(enumerate(filepath_batches),
                           bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}',
                           total=len(filepath_batches)):
        with pool.Pool(NUM_OF_CORES) as p:
            batch = p.map(pd.read_csv, batch)
        for file_df in batch:
            base_df = pd.concat([base_df, file_df])

    if not os.path.exists(path_to_result):
        os.makedirs(path_to_result)

    print(f"Сохранение объединённой таблицы")
    with open(os.path.join(path_to_result, f"main_df.pkl"),
              'wb') as handle:
        pickle.dump(base_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return base_df

def preprocess_data(main_df: pd.DataFrame = None):
    print(f"Очистка колонок")
    if main_df is None:
        with open(os.path.join(path_to_result, f"main_df.pkl"),
                  "rb") as handle:
            main_df = pickle.load(handle)

    main_df = main_df.reset_index()
    main_df = main_df.drop(columns=["index", "Unnamed: 0"])

    # Количество пустых значений в каждой колонке
    na_by_columns = dict(
        sorted({column: main_df[column].isna().sum() for column in
                main_df.columns}.items(), key=lambda x: -x[1]))

    for column, vals in tqdm(na_by_columns.items(),
                             bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                             total=len(na_by_columns)):
        try:
            # Попытка обработать колонку как числовую
            condition = (column in FORBIDDEN_COLUMNS or
                         vals > 0.1 * main_df.shape[0] or
                         np.std(main_df[column]) == 0)
            if condition:
                # Дроп лишних колонок
                main_df = main_df.drop(columns=[column])
            else:
                # Обработка остальных
                lb = main_df[column].quantile(0.005)
                ub = main_df[column].quantile(0.995)
                main_df['bad_val'] = (main_df[column] > lb) & (
                            main_df[column] < ub)
        except:
            # Колонка не числовая, добавляем тег
            main_df = main_df.rename({column: f"{column}_non_number"})
    main_df = main_df[main_df['bad_val'] == True]
    main_df = main_df.dropna()

    print(f"Сохранение очищенной таблицы")
    with open(os.path.join(path_to_result, f"main_df_cleared.pkl"),
              "wb") as handle:
        pickle.dump(main_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return main_df


def create_target_df(target: str, main_df: pd.DataFrame = None,
                     kbest_count: int = 200, rfe_count: int = 50):

    if main_df is None:
        with open(os.path.join(path_to_result, f"main_df_cleared.pkl"),
                  "rb") as handle:
            main_df = pickle.load(handle)

    main_df = main_df.drop(columns=["bad_val", "billet_id", "WBF_sgl_CHARGING_TIME", "WBF_sgl_CROSSING_TIME_SOA",
                                    "WBF_sgl_DISCHARGING_TIME", "WBF_sgl_IS_DISCHARGED", "WBF_sgl_MARKA",
                                    "WBF_sgl_MELT_NUMBER", "WBF_sgl_ROLLING_PROFILE", "WBF_sgl_STEEL_MARK",
                                    "WBF_sgl_TIME_IN", "WBF_sgl_TIME_OUT", "WBF_sgl_VALIDITY", "billet_number"])

    main_df['LNK100_Torsion_ median_075-3_T_abs'] = main_df[target].abs()
    target = 'LNK100_Torsion_ median_075-3_T_abs'

    print(f"Выбор оптимальных фичей")
    y = main_df[target]
    X = main_df[[column for column in main_df.columns if 'LNK100' not in column]]
    selector = SelectKBest(score_func=f_regression, k=kbest_count)
    _ = selector.fit_transform(X, y)
    X_kbest = main_df[list(selector.get_feature_names_out())]

    model = catboost.CatBoostRegressor()
    rfe = RFE(model, n_features_to_select=rfe_count, verbose=0)
    _ = rfe.fit_transform(X_kbest, y)
    target_df = main_df[[*list(rfe.get_feature_names_out()), target]]
    print(f"Сохранение очищенной таблицы")
    with open(os.path.join(path_to_result, f"{target}_df.pkl"), "wb") as handle:
        pickle.dump(target_df, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # base_df = concat_files()
    # clr_df = preprocess_data()
    # create_target_df('LNK100_Torsion_[i_103.0]_[-3_end]_[max]')
    # create_target_df('LNK100_Torsion_[i_103.0]_[-3_end]_[mean]')
    create_target_df('LNK100_Torsion_[i_103.0]_[-3_end]_[median]')
    # create_target_df('LNK100_Torsion_[i_103.0]_[-3_end]_[min]')
