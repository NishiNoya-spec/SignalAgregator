import pandas as pd
import numpy as np
from glob import glob

import sklearn.model_selection
from tqdm import tqdm
import os
import pickle
import math
from multiprocessing import cpu_count, pool
from sklearn.neighbors import LocalOutlierFactor
import re
from dto import PreprocessResult
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression, RFECV
import catboost

ABSPATH = r"Z:\DS\filtered_std_new\2023-03-27_2023-11-08"
FORBIDDEN_COLUMNS = [
    'Unnamed: 0', 'billet_id', 'billet_number',
    'WBF_sgl_CROSSING_TIME_CNV', 'WBF_sgl_CROSSING_TIME_PHE', 'WBF_sgl_CROSSING_TIME_HE1', 'WBF_sgl_CROSSING_TIME_HE2',
    'WBF_sgl_HEIGHT_IN', 'WBF_sgl_HEIGHT_OUT', 'WBF_sgl_WIDTH_IN', 'WBF_sgl_WIDTH_OUT', 'WBF_sgl_PZAG',
    'WBF_sgl_MARKA_PROCESSED',  'WBF_sgl_ATTESTATED', 'WBF_sgl_FURNACE_FK', 'WBF_sgl_TOT_X_NODE', 'WBF_sgl_TOT_Y_NODE',
    'WBF_sgl_TOT_Z_NODE', 'WBF_sgl_IS_DISCHARGED', 'WBF_sgl_H_WBF', 'WBF_sgl_DENSITY', 'WBF_sgl_TEMPERATURE_IN',
    'WBF_sgl_STATUS', 'bd1.descaling.ready', 'bd1.descaling.pressure.on', 'bd2.dk122-bs-01.tilter_4_angle.set',
    'ur.reference_inclination_horizontal_top_roll', 'WBF_sgl_Z1_LV2_IN_percent', 'WBF_sgl_Z2_LV2_IN_percent',
    'WBF_sgl_Z3_LV2_IN_percent', 'WBF_sgl_Z4_LV2_IN_percent', 'bd1.dr21.bfz01.sensor_point', 'WBF_sgl_TIME_IN',
    'WBF_sgl_TIME_OUT', 'WBF_sgl_CHARGING_TIME', 'WBF_sgl_DISCHARGING_TIME', 'WBF_sgl_PIECE_PK',
    'WBF_sgl_CROSSING_TIME_SOA', 'WBF_sgl_MELT_NUMBER', 'WBF_sgl_STEEL_MARK']
FORBIDDEN_COLUMNS_2 = pd.read_csv('forbidden.csv')['column'].to_list()
CATEGORICAL_KEYS = [
    'WBF_sgl_STRATEGY', 'WBF_sgl_CHG_TEMP', 'WBF_sgl_VALIDITY', 'WBF_sgl_ROLLING_PROFILE', 'WBF_sgl_TEMPERATURE_OUT',
    'WBF_sgl_CONTROL', 'WBF_sgl_PROFILE_VALID_ID', 'WBF_sgl_BISRA', 'WBF_sgl_PLA_PACE_RATE', 'WBF_sgl_TARGET_TEMP',
    'WBF_sgl_DCHG_TARGET_TEMP_PDI', 'WBF_sgl_STRAND_NUMBER', 'WBF_sgl_MARKA', 'WBF_sgl_STEEL_CODE',
    'WBF_sgl_STEEL_MARK', 'WBF_sgl_STRAND_BILLET_NUMBER', 'WBF_sgl_W_WBF', 'WBF_sgl_LENGTH_ORDERED',
    'WBF_sgl_WEIGHT_ORDERED', 'WBF_sgl_WIDTH_PROCESSED']
IGNORED_COLUMNS = [
    'bad_columns'
]
NUM_OF_CORES = 19
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
        with open(os.path.join(path_to_result, f"cleaned_df.pkl"), "rb") as handle:
            data_df = pickle.load(handle)
        with open(os.path.join(path_to_result, f"np_data.pkl"), "rb") as handle:
            data_dict = pickle.load(handle)
    mark = []
    for val in data_dict['WBF_sgl_MARKA']:
        if val == '90ХАФ_Р65' or val == 'Э90ХАФ_Р65':
            class_mark = 1
        elif val == 'Э76ХФ_Р65' or val == '76ХФ_Р65':
            class_mark = 0
        else:
            class_mark = -1
        mark.append(class_mark)
    data_df['WBF_sgl_MARKA'] = mark
    del data_dict
    data_df = data_df[data_df['WBF_sgl_MARKA'] != -1]
    data_df = data_df[data_df['WBF_sgl_ROLLING_PROFILE'] == 1]
    data_df = data_df[data_df['to_del'] == 0]
    with open(os.path.join(path_to_result, f"double_cleaned_df.pkl"), "wb") as handle:
        pickle.dump(data_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    a = 1
    # data_dict['WBF_sgl_TIME_DELTA'] = np.array([(pd.Timestamp(out_t) - pd.Timestamp(in_t)).seconds for in_t, out_t in
    #                                             zip(data_dict['WBF_sgl_TIME_IN'], data_dict['WBF_sgl_TIME_OUT'])])
    # data_dict['WBF_sgl_CHARGING_DELTA'] = np.array(
    #     [(pd.Timestamp(out_t) - pd.Timestamp(in_t)).seconds for in_t, out_t in
    #      zip(data_dict['WBF_sgl_CHARGING_TIME'], data_dict['WBF_sgl_DISCHARGING_TIME'])])
    # data_dict['WBF_sgl_VALIDITY'] = np.array([1 if val == 'N' else 0 for val in data_dict['WBF_sgl_VALIDITY']])
    # data_dict['WBF_sgl_ROLLING_PROFILE'] = np.array([1 if val == 'Р65' else 0
    #                                                  for val in data_dict['WBF_sgl_ROLLING_PROFILE']])
    # mark = []
    # for val in data_dict['WBF_sgl_MARKA']:
    #     if val == '90ХАФ_Р65' or val == 'Э90ХАФ_Р65':
    #         class_mark = 1
    #     elif val == 'Э76ХФ_Р65' or val == '76ХФ_Р65':
    #         class_mark = 0
    #     else:
    #         class_mark = -1
    #     mark.append(class_mark)
    # data_dict['WBF_sgl_MARKA'] = np.array(mark)
    # with open(os.path.join(path_to_result, f"preprocess_otl_2.pkl"), "rb") as handle:
    #     data_dict['to_del'] = np.array(pickle.load(handle))
    # dict_keys = list(data_dict.keys())
    # for colname in dict_keys:
    #     if colname in FORBIDDEN_COLUMNS_2:
    #         del data_dict[colname]
    #     elif sum([1 for key in FORBIDDEN_COLUMNS if key in colname]) > 0:
    #         del data_dict[colname]
    # df = pd.DataFrame()
    # dict_keys = list(data_dict.keys())
    # for column in dict_keys:
    #     df[column] = data_dict[column]
    #     del data_dict[column]
    # with open(os.path.join(path_to_result, f"cleaned_df.pkl"), "wb") as handle:
    #     pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # cut_len = math.ceil(len(data_dict)/2)
    # number = 2
    # data_dict = dict(tuple(data_dict.items())[cut_len * (number - 1):cut_len*number])
    # # Количество пустых значений в каждой колонке
    # na_by_columns = dict(
    #     sorted({column: pd.Series(array).isna().sum()
    #             for column, array in
    #             data_dict.items()}.items(), key=lambda x: x[1]))
    #
    # inputs = [(column, data_dict[column], na_by_columns[column]) for column in na_by_columns.keys()]

    # DEBUG
    # col = 'U0_u0.descaling.pressure.extrapolated_[i_103.0]_[0_3]_[median]'
    # result = [_preprocess((col, data_dict[col], na_by_columns[col]))]
    # for inp in inputs:
    #     print('pupa')
    #     _preprocess(inp)

    with pool.Pool(NUM_OF_CORES) as p:
        result = list(tqdm(p.imap(_preprocess, inputs),
                      bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                      total=len(na_by_columns)))

    df = []
    # to_delete = [0 for _ in range(24167)]
    with open(os.path.join(path_to_result, f"preprocess_otl_1.pkl"), "rb") as handle:
        to_delete = pickle.load(handle)
    del data_dict
    for res in result:
        df.append({"Name": res.name, "To Drop": res.to_drop, "To Rename": res.to_rename,
                   "Count Outliers": res.count_outliers, "Count Unique": res.count_unique})
        if not res.to_rename and not res.to_drop:
            new = [res.outliers[i] + to_delete[i] for i in range(24167)]
            to_delete = new.copy()
    print(sum([1 for i in to_delete if i == 0]))
    del result
    df = pd.DataFrame(df)
    df.to_excel(f"result_{number}.xlsx")
    with open(os.path.join(path_to_result, f"preprocess_otl_{number}.pkl"), "wb") as handle:
        pickle.dump(to_delete, handle, protocol=pickle.HIGHEST_PROTOCOL)
    data_len = len(data_dict)
    all_bad_values = [0 for _ in range(data_len)]
    for ft in result:
        if ft.to_rename:
            data_dict[f"STR__{ft.name}"] = data_dict.pop(ft.name)
        else:
            if ft.to_drop:
                del data_dict[ft.column]
            else:
                all_bad_values = all_bad_values + ft.outliers
        print(f"{ft.name}: {sum(ft.outliers)} / {data_len - np.count_nonzero(all_bad_values == 0)} / {data_len}")
        a = 1

    main_df = main_df[main_df['bad_val'] == True]
    main_df = main_df.dropna()

    print(f"Сохранение очищенной таблицы")
    with open(os.path.join(path_to_result, f"main_df_cleared.pkl"),
              "wb") as handle:
        pickle.dump(main_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path_to_result, f"cols_to_drop.pkl"),
              "wb") as handle:
        pickle.dump(drop_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return main_df


def _unique_vals(data: tuple):
    column, array = data
    re_pattern = re.match(r'[\w.,_-]+\[[\w.]+]', column)
    key_name = re_pattern.group(0) if re_pattern else column
    unique_values = []
    for x in array:
        if type(x) is str:
            unique_values.append(x)
        elif ~np.isnan(x):
            unique_values.append(x)
    del column, array, data
    return key_name, set(unique_values)


def _preprocess(data: tuple) -> PreprocessResult:
    column, array, nans = data
    if True in [col in column for col in IGNORED_COLUMNS]:
        result = PreprocessResult(column, False, False, list(np.zeros(len(array))), 0, 0)
    elif True in [col in column for col in FORBIDDEN_COLUMNS]:
        result = PreprocessResult(column, False, True, [], 0, 0)
    else:
        try:
            # Попытка обработать колонку как числовую
            condition = (nans > 0.1 * len(array) or
                         array.std() == 0)
            if condition:
                # Дроп лишних колонок
                result = PreprocessResult(column, False, True, [], 0, 0)
            else:
                if True in [col in column for col in CATEGORICAL_KEYS] or len(np.unique(array)) <= 20:
                    # Обраотка категориальных сигналов
                    outliers = np.zeros(len(array))
                    outliers[np.where(np.isnan(array))] = 1
                    result = PreprocessResult(column, False, False, list(outliers),
                                              outliers.sum(), len(np.unique(array)))
                else:
                    # Обработка нормальных сигналов
                    # no_nan_array = array[~np.isnan(array)]
                    # lof = LocalOutlierFactor(n_neighbors=10, algorithm="ball_tree")
                    # otl_values = [no_nan_array[idx] for idx, val in
                    #               enumerate(lof.fit_predict(no_nan_array.reshape(-1, 1))) if val != 1]
                    # outliers = [1 if val in otl_values or np.isnan(val) else 0 for val in array]
                    outliers = [1 if np.isnan(val) else 0 for val in array]
                    result = PreprocessResult(column, False, False, list(outliers),
                                              sum(outliers), len(np.unique(array)))
        except:
            # Колонка не числовая, добавляем тег
            result = PreprocessResult(column, True, False, [], len(set([x for x in array if type(x) is str])), 0)
    return result


def create_target_df(target: str, main_df: pd.DataFrame = None,
                     kbest_count: int = 5000, rfe_count: int = 1000):

    if main_df is None:
        with open(os.path.join(path_to_result, f"main_df_cleared.pkl"),
                  "rb") as handle:
            main_df = pickle.load(handle)
    # main_df = main_df.drop(columns=["bad_columns"])
    # print(f"Выбор оптимальных фичей")
    y = main_df[target]
    X = main_df[[column for column in main_df.columns
                 if 'LNK100' not in column and "non_number" not in column]]
    # X_names = X.columns
    # y = y.to_numpy()
    # X = X.to_numpy()
    # del main_df
    # selector = SelectKBest(score_func=f_regression, k=kbest_count)
    # _ = selector.fit_transform(X, y)
    # top_names = [X_names[int(idx[1:])] for idx in list(selector.get_feature_names_out())]
    # with open(os.path.join(path_to_result, f"top_names.pkl"), "wb") as handle:
    #     pickle.dump(top_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # X_kbest = main_df[list(selector.get_feature_names_out())]
    # print('Сохранение полувыборки')
    # with open(os.path.join(path_to_result, f"x_best.pkl"), "wb") as handle:
    #     pickle.dump(main_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model = catboost.CatBoostRegressor()
    rfe = RFECV(model, step = 10, min_features_to_select=rfe_count, verbose=0, n_jobs=10)
    _ = rfe.fit_transform(X, y)
    target_df = main_df[[*list(rfe.get_feature_names_out()), target]]
    print(f"Сохранение очищенной таблицы")
    with open(os.path.join(path_to_result, f"{target}_df.pkl"), "wb") as handle:
        pickle.dump(target_df, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    with open(os.path.join(path_to_result, f"otk_df.pkl"), "rb") as handle:
        data_dict = pickle.load(handle)
    data_dict = data_dict.drop(columns=['bad_columns', 'billet_id'])

    vals = []
    for val in zip(data_dict["A"], data_dict["B"], data_dict["C"], data_dict["Y"]):
        if sum(val) > 0:
            vals.append(1)
        else:
            vals.append(0)
    data_dict["ABCY"] = vals

    vals = []
    for val in zip(data_dict["A"], data_dict["Y"]):
        if sum(val) > 0:
            vals.append(1)
        else:
            vals.append(0)
    data_dict["AY"] = vals

    importances = {}
    for key in ["A", "B", "C", "Y", "ABCY", "AY"]:
        class_one = data_dict[data_dict[key] == 1].sample(frac=1).reset_index( drop=True)
        class_zero = data_dict[data_dict[key] == 0].sample(frac=1).reset_index(drop=True)
        min_size = min(len(class_one), len(class_zero))
        train_data = pd.concat([class_zero.iloc[:min_size, ], class_one.iloc[:min_size, ]])
        cbt = catboost.CatBoostClassifier(iterations=2000, depth=6)
        X = train_data.drop(columns=["A", "B", "C", "Y", "ABCY", "AY"])
        y = train_data[key]
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, stratify=y, test_size=0.1)
        train = catboost.Pool(X_train, y_train)
        test = catboost.Pool(X_test, y_test)
        cbt.fit(train, eval_set=test, verbose=2)
        ft_imp = dict(sorted({list(X_train.columns)[idx]: imp for idx, imp
                              in enumerate(cbt.get_feature_importance(data=test, type='LossFunctionChange'))}.items(),
                             key=lambda x: -x[1]))
        importances[key] = dict(list(ft_imp.items())[:50])
    with open(os.path.join(path_to_result, f"balanced_otk_importances.pkl"), "wb") as handle:
        pickle.dump(importances, handle, protocol=pickle.HIGHEST_PROTOCOL)

    good_idxs = []
    for idx, billet in enumerate(data_dict["billet_id"]):
        if billet in main_df["billet_id"]:
            good_idxs.append(idx)
    for key in ["A", "B", "C", "Y", "billet_id"]:
        data_dict[key] = np.array(data_dict[key])[good_idxs]
        print(len(data_dict[key]))

    df = pd.DataFrame()
    dict_keys = list(data_dict.keys())
    for column in tqdm(dict_keys):
        df[column] = data_dict[column]
        del data_dict[column]
    with open(os.path.join(path_to_result, f"otk_df.pkl"), "wb") as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    a = 1
