import os
import pandas as pd


def create_save_path(path_to_base_dir, new_dir_name) -> str:
    new_dir_path = os.path.join(path_to_base_dir, new_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)
    return new_dir_path


def get_time_period_in_str(time_period: pd.Series):
    processing_dates = time_period.dt.date
    return str(processing_dates.min()) + "_" + str(processing_dates.max())
