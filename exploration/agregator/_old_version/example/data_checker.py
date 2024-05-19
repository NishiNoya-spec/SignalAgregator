import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments_aggregation import data_mapping, dto
from experiments_aggregation.aggregator import AggregationFactory
from experiments_aggregation.source_parser import parse_settings

AGREGAT_LENGTH = {"U0": 106}
DELTA = 5


@dataclass
class AgregatChecker:
    agregat_name: str
    limits_path: str = r"D:\zharikova_ep\cv_projects\9684_dev\exploration\experiments_aggregation\example\u0_stats.csv"
    limits: Dict = field(default_factory=dict)
    aggregators: AggregationFactory = AggregationFactory()

    def __post_init__(self):
        limits = pd.read_csv(self.limits_path, sep=";", decimal=",").dropna()
        self.limits = dict()
        for signal, data in limits.groupby('Исходный сигнал'):
            self.limits[signal] = data.drop(columns=['Исходный сигнал']).set_index('Производный сигнал').to_dict("index")

    def check(self, signal_values: np.ndarray, signal_points: np.ndarray, signal_name: str):

        is_std_zero = self.is_std_zero(signal_values)
        limits_report = self.check_limits(signal_values, signal_points, signal_name)

        return is_std_zero, limits_report

    def has_all_points(self, signal_points) -> bool:
        min_limit = 0 <= np.min(signal_points) <= 0.5
        max_limit = (AGREGAT_LENGTH[self.agregat_name] - DELTA) <= np.max(signal_points) <= (AGREGAT_LENGTH[self.agregat_name] + DELTA)
        return min_limit and max_limit

    def is_std_zero(self, signal_values) -> bool:
        return np.std(signal_values) == 0

    def check_limits(self, signal_values: np.ndarray, signal_points: np.ndarray, signal_name: str) -> List[List[Optional[float]]]:

        limits = self.limits.get(signal_name, None)
        if limits is None:
            return [[signal_name, None]]

        signal = pd.DataFrame({'point': signal_points, 'signal': signal_values})

        limits_report = list()
        for agg_method, slice_info in limits.items():
            if agg_method == "tg":
                continue
            slice_range = slice_info['Отрезок на рельсе']
            min_value = slice_info['min']
            max_value = slice_info['max']
            start_point = "start"
            end_point = "end"

            if slice_range == "all":
                signal_slice = signal
            else:
                start_point, end_point = slice_range.split("_")
                signal_slice = signal[(signal['point'] <= start_point) & (signal['point'] >= end_point)]

            agg_value = self.aggregators.agg_by_method(signal_slice["signal"], agg_method)

            min_limit_approved = agg_value >= min_value
            max_limit_approved = agg_value <= max_value

            if min_limit_approved and max_limit_approved:
                continue
            else:
                limits_report.append([agg_method, start_point, end_point, min_limit_approved, max_limit_approved])

        return limits_report


def check_data(mapping):

    general_report = GeneralReport()

    for billet_id, billet_map in tqdm(mapping.items()):
        not_enough_points_aggregates = []
        none_signals = dict()
        std_zero_signals = dict()
        out_of_limits_signals = dict()
        aggregate_checker = AgregatChecker("U0")
        for aggregate, file_path in billet_map.items():
            if aggregate == "U0":
                data = pd.read_csv(file_path, sep=";", decimal=",")

                general_report.append_time(os.path.split(file_path)[-1].split("_")[0])
                general_report.append_billet_id(billet_id)


                has_all_points = aggregate_checker.has_all_points(data["billet_points"])

                if has_all_points:
                    for signal_name in data.columns:
                        if signal_name not in ["moment", "billet_points"]:
                            current_data = data[["billet_points", signal_name]].dropna()
                            if len(current_data) > 0:
                                is_std_zero, limits_reports = aggregate_checker.check(
                                    np.array(current_data[signal_name]),
                                    np.array(current_data["billet_points"]),
                                    signal_name,
                                )

                                if is_std_zero:
                                    append_aggregate_data(std_zero_signals, aggregate, signal_name)

                                for limits_report in signals_limits_description(limits_reports, signal_name):
                                    append_aggregate_data(out_of_limits_signals, aggregate, limits_report)
                            else:
                                append_aggregate_data(none_signals, aggregate, signal_name)


                else:
                    not_enough_points_aggregates.append(aggregate)

        general_report.append_general_report(not_enough_points_aggregates, "full_points", f"There are not enough points for {', '.join(not_enough_points_aggregates)}")
        general_report.append_general_report(std_zero_signals, "std_zero_signals", f"std = 0:  {std_zero_signals}")
        general_report.append_general_report(out_of_limits_signals, "signals_limits", f"{out_of_limits_signals}")
        general_report.append_general_report(none_signals, "none_signals", f"{none_signals}")

    return general_report.dataframe


def signals_limits_description(limits_reports, signal_name):
    limits_report = []
    for limits_report in limits_reports:
        if limits_report[1] is None:
            limits_report.append(f"{signal_name} has no limits")
        else:
            agg_method, start_point, end_point, min_limit_approved, max_limit_approved = limits_report
            limits_report.append(f"{signal_name}_{start_point}_{end_point}_{agg_method} " \
                          f"min_limit_approved={min_limit_approved} " \
                          f"max_limit_approved={max_limit_approved}")

    return limits_report


def append_aggregate_data(data, aggregate, value):
    if aggregate in data:
        data[aggregate].append(value)
    else:
        data[aggregate] = [value]


class GeneralReport:
    general_report: Dict[str, List[str]] = {
        "start_rolling_time": [],
        "billet_id": [],
        "full_points": [],
        "none_signals": [],
        "std_zero_signals": [],
        "signals_limits": [],
    }

    @property
    def dataframe(self):
        return pd.DataFrame(self.general_report)

    def append_time(self, start_time):
        self.general_report["start_rolling_time"].append(
            datetime.strptime(start_time, '%Y%m%d%H%M%S'))

    def append_billet_id(self, billet_id):
        self.general_report["billet_id"].append(billet_id)

    def append_general_report(self, report, report_name, value):
        if len(report):
            self.general_report[report_name].append(value)
        else:
            self.general_report[report_name].append(None)


def get_mapping():
    constants = dto.Constants(
        PATH_TO_RESULT=r"\\ZSMK-9684-001\Data\DS\first_3_m_norm_test",
        NUM_OF_CORES=15,
        MIN_TIME=datetime(year=2023, month=1, day=1, hour=0, minute=0, second=0),
        MARK_FILTER=True,
        MARK=['76ХФ', '90ХАФ'],
        PATH_TO_METADATA=os.path.join(r'\\ZSMK-9684-001\Data\DS',
                                      "metadata/meta_1408_1608*.xlsx"),
        METADATA_BILLET_ID="BilletId"
    )

    settings = pd.read_excel(
        "D:\zharikova_ep\cv_projects\9684_dev\exploration\experiments_aggregation\example\settings.xlsx", index_col=[0],
        header=None).iloc[:, :-1]
    sources_settings = parse_settings(settings)

    print("Подготовка данных. Создание мапинга...")

    creator = data_mapping.MappingCreator(constants=constants)
    metadata = creator.get_metadata()
    mapping, report = creator.create_mapping(metadata=metadata,
                                             settings=sources_settings)
    return mapping, report


if __name__ == "__main__":
    mapping, report = get_mapping()

    check_data(mapping).to_csv("report.csv", sep=";", decimal=",")