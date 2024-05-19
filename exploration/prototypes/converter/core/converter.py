import os
from datetime import datetime

import numpy as np
import pandas as pd

from exploration.prototypes.converter.constants import BILLET_POINT

PATH_TO_DATA = r"/app"

AGG_METHODS = {
    "Среднее": "mean",
    "Медиана": "median",
    "Стандартное отклонение": "std",
    "Минимум": "min",
    "Максимум": "max",
    "Дисперсия": "var",
    "Асимметрия": "skew",
    "Эксцесс": "kurtosis",
    "Тангенс": "tg"
}

RAILE_ZONE = {"Голова": "Head", "Хвост": "Tail", "Весь рельс": "All"}


class Converter:

    def __init__(self):
        self.files_count = 0
        self.processed_files = 0

    def convert_data(
        self, filter_flag, abs_flag, data_paths, signals, methods, zones,
        zones_min_len, zones_max_len, result_path
    ):
        count_aggregation_patterns = len(filter_flag)
        result = pd.DataFrame()
        self.files_count = len(data_paths)
        for i, path in enumerate(data_paths):
            print(f"Processing {path} ...")

            row = self._get_result_row(
                i + 1, path, count_aggregation_patterns, filter_flag, abs_flag,
                signals, methods, zones, zones_min_len, zones_max_len
            )

            self.processed_files += 1
            print("Adding result...")
            result = result.append(row, ignore_index=True)
        result.to_csv(
            os.path.join(result_path, "result.csv"),
            sep=";",
            decimal=",",
            encoding="windows-1251",
            index=False
        )
        return result

    def _get_result_row(
        self, file_number, path, count_aggregation_patterns, filter_flag,
        abs_flag, signals_dropdown, methods_dropdown, zones_dropdown,
        zones_min_len, zones_max_len
    ):

        # когда формирование файлов будет происходить из текущей системы
        file_name = os.path.split(path)[-1]

        cut_billet = pd.read_csv(path, sep=";", decimal=",").fillna(0)
        # Временное решение разночтений в именовании времени.

        if "billet_points" in cut_billet.columns:
            cut_billet = cut_billet.rename(
                columns={"billet_points": BILLET_POINT}
            )

        if "BilletPoints" in cut_billet.columns:
            cut_billet = cut_billet.rename(
                columns={"BilletPoints": BILLET_POINT}
            )

        row = self._generate_row(
            file_number, file_name, path, cut_billet[BILLET_POINT].max()
        )

        for j in range(count_aggregation_patterns):
            use_filter = len(filter_flag[j]) > 0
            use_abs = len(abs_flag[j]) > 0
            for_billet_head = "H" in zones_dropdown[j]
            billet_meters = [zones_min_len[j], zones_max_len[j]]
            billet_meters.sort()

            selected_signals = signals_dropdown[j]
            method = methods_dropdown[j]
            dataframe_of_selected_signals = cut_billet[selected_signals
                                                       + [BILLET_POINT]]
            aggregated_data = (
                self._get_aggregated_data_of_billet_piece_for_signals(
                    method,
                    cut_billet=dataframe_of_selected_signals,
                    start_meter=billet_meters[0],
                    end_meter=billet_meters[1],
                    for_billet_head=for_billet_head,
                    use_filter=use_filter,
                    use_abs=use_abs,
                )
            )
            for signal in selected_signals:
                column_name = self._generate_column_name(
                    signal, method, use_filter, use_abs, row['Agregat'],
                    row['Prohod'], zones_dropdown[j], billet_meters
                )
                row[column_name] = aggregated_data[signal]
        return row

    def _generate_row(self, file_number, file_name, path, rail_length):
        row = dict()

        split_file_name = file_name.split("_")
        row["N File"] = file_number
        row["FileName"] = file_name
        row["ID"] = split_file_name[-4]
        row["DateTime"] = datetime.strptime(
            split_file_name[-5],
            '%Y%m%d%H%M%S',
        )
        row["Length"] = rail_length
        row["Agregat"] = split_file_name[-3]
        row["Prohod"] = split_file_name[-2]
        return row

    def _generate_column_name(
        self, signal, method, use_filter, use_abs, agregat, prohod, zone_info,
        billet_meters
    ):
        rail_part = '' if 'All' in zone_info \
            else f"{zone_info[0]}.{billet_meters[0]}-{billet_meters[1]}m."

        return f"{agregat}({prohod}).{signal}.{rail_part}{method}" \
               f"{'.f' * use_filter}{'.abs' * use_abs}"

    def _get_aggregated_data_of_billet_piece_for_signals(
        self,
        method,
        cut_billet: pd.DataFrame,
        start_meter=0,
        end_meter=6,
        for_billet_head=True,
        use_filter=True,
        use_abs=True,
    ):

        using_billet_points = cut_billet[BILLET_POINT] \
            if for_billet_head \
            else cut_billet[BILLET_POINT].max() - cut_billet[BILLET_POINT]

        signals_of_first_billet_n_meters = cut_billet[
            (using_billet_points <= end_meter)
            & (using_billet_points >= start_meter)]
        signals_of_first_billet_n_meters = self._get_filtered_data(
            signals_of_first_billet_n_meters) \
            if use_filter else signals_of_first_billet_n_meters
        signals_of_first_billet_n_meters = (
            signals_of_first_billet_n_meters.abs()
            if use_abs else signals_of_first_billet_n_meters
        )
        return self._get_aggregated_value_for_signal(
            method, signals_of_first_billet_n_meters
        )

    def _get_aggregated_value_for_signal(
        self, method, cut_billet: pd.DataFrame
    ):
        if method == "tg":
            return self._get_tg_value_for_signals(cut_billet)
        return getattr(cut_billet, method)()

    def _get_tg_value_for_signals(self, cut_billet: pd.DataFrame):
        tg_data = dict()
        for signal in cut_billet.columns:
            if signal != BILLET_POINT:
                cut_billet_without_na = cut_billet[[signal,
                                                    BILLET_POINT]].dropna()
                Ox = cut_billet_without_na[BILLET_POINT]
                Oy = cut_billet_without_na[signal]
                A = np.c_[Ox, np.ones(len(Oy))]
                p, residuals, rank, svals = np.linalg.lstsq(A, Oy, rcond=None)
                dots = A.dot(p)
                if len(dots) >= 2:
                    y = A.dot(p)[1] - A.dot(p)[0]
                    x = Ox.iloc[1] - Ox.iloc[0]
                    tg_data[signal] = y / x if x != 0 else y / 0.00001
                else:
                    tg_data[signal] = "No tg"
        return tg_data

    def _get_filtered_data(self, cut_billet: pd.DataFrame):
        billet_points = cut_billet[BILLET_POINT]
        median = cut_billet.median()
        std = cut_billet.std()
        min_limit = median - 3 * std
        max_limit = median + 3 * std
        filtered_data = cut_billet.mask(
            (cut_billet > max_limit | cut_billet < min_limit),
            other=None,
            inplace=False,
            axis=None,
            level=None,
            errors='raise',
            try_cast=None
        )
        filtered_data[BILLET_POINT] = billet_points
        return filtered_data
