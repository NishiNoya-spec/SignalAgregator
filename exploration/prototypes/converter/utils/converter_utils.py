# -*- coding: utf-8 -*-
import base64
import io
import os
import zipfile
from glob import glob
from os.path import join

from exploration.prototypes.converter.converter_templates import (
    ConverterInterfaceElementsGenerator,
)
from exploration.prototypes.converter.core.converter import (
    PATH_TO_DATA,
    Converter,
)
from exploration.prototypes.converter.dash_templates import (
    InterfaceElementsGenerator,
)

interface_elements_generator = InterfaceElementsGenerator()
convertor_interface_elements_generator = ConverterInterfaceElementsGenerator()

convertor = Converter()


def rail_zones_field_interface(selected_zones, min_values, max_values):
    min_len = 0
    max_len = 150
    disabled_values = []
    for i, selected_zone in enumerate(selected_zones):
        if selected_zone is None:
            disabled_values.append(False)

        elif "All" in selected_zone:
            disabled_values.append(True)
            min_values[i] = min_len
            max_values[i] = max_len
        else:
            disabled_values.append(False)

        if min_values[i] is None:
            min_values[i] = min_len
            max_values[i] = max_len

    return [min_values, max_values, disabled_values, disabled_values]


def buttons_disabler(
    signals_dropdown, methods_dropdown, zones_dropdown, zones_min_len,
    zones_max_len
):
    calculator_button = True
    if zones_max_len is None:
        return [True, True]

    for row in range(len(zones_max_len)):
        if signals_dropdown[row] is None or \
                methods_dropdown[row] is None or \
                zones_dropdown[row] is None or \
                zones_min_len is None or \
                zones_max_len is None or \
                zones_min_len == zones_max_len:

            calculator_button = True
        else:
            calculator_button = False

    return [calculator_button, calculator_button]


def save_uploaded_data(content, name, input_data_path):
    _, content_string = content.split(',')
    content_decoded = base64.b64decode(content_string)
    zip_str = io.BytesIO(content_decoded)
    zip_obj = zipfile.ZipFile(zip_str, 'r')
    zip_obj.extractall(PATH_TO_DATA, )
    folder = name.replace(".zip", "")
    os.rename(
        os.path.join(PATH_TO_DATA, folder),
        os.path.join(PATH_TO_DATA, input_data_path)
    )
    for file_path in glob(join(PATH_TO_DATA, input_data_path, "*.csv")):
        try:
            os.renames(file_path, file_path.encode('cp437').decode('utf-8'))
        except UnicodeDecodeError:
            os.renames(file_path, file_path.replace("ï", "Л"))

    data_path = join(
        PATH_TO_DATA,
        input_data_path,
        "*.csv",
    )
    interface_elements_generator. \
        set_data_paths_and_signal_names_for_aggregation(data_path)


def set_set_data_paths_and_signal_names_for_aggregation(data_path):
    data_path = join(
        data_path,
        "*.csv",
    )
    convertor_interface_elements_generator. \
        set_data_paths_and_signal_names_for_aggregation(data_path)


def convert(
    filter_flag, abs_flag, data_paths, signals_dropdown, methods_dropdown,
    zones_dropdown, zones_min_len, zones_max_len, result_path
):
    convertor.convert_data(
        filter_flag, abs_flag, data_paths, signals_dropdown, methods_dropdown,
        zones_dropdown, zones_min_len, zones_max_len, result_path
    )
