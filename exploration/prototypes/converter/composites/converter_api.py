import os
import shutil
from os.path import join

import dash
import pandas as pd
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output
from flask import Flask

from exploration.prototypes.converter.constants import (
    CONVERTOR_INPUT_DATA,
    CONVERTOR_OUTPUT_DIR,
)
from exploration.prototypes.converter.core.converter import PATH_TO_DATA
from exploration.prototypes.converter.dash_templates import (
    CONVERTOR_INTERFACE,
    CONVERTOR_PAGE_TEMPLATE,
    STATUS_PAGE,
)
from exploration.prototypes.converter.utils import converter_utils
from exploration.prototypes.converter.utils.dash_input_output_params import (
    ConverterInputOutputStates,
)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = Flask("sensor_to_billet_matching", template_folder="templates")

app_convertor = dash.Dash(
    "EvrazConvertor",
    external_stylesheets=external_stylesheets,
    prevent_initial_callbacks=True,
    server=server,
    url_base_pathname='/'
)

app_status = dash.Dash(
    "EvrazStatus",
    external_stylesheets=external_stylesheets,
    prevent_initial_callbacks=True,
    server=server,
    url_base_pathname='/status/'
)

app_convertor.layout = CONVERTOR_PAGE_TEMPLATE

app_status.layout = STATUS_PAGE

convertor_in_out_states = ConverterInputOutputStates()

#######################
# Convertor callbacks #
#######################


@app_convertor.callback(
    convertor_in_out_states.dropdown_output,
    convertor_in_out_states.dropdown_input,
    convertor_in_out_states.dropdown_state
)
def display_dropdowns(n_clicks, children):
    children.append(
        converter_utils.interface_elements_generator.generate_interface_row(
            index=n_clicks
        )
    )
    return children


@app_convertor.callback(
    convertor_in_out_states.rail_zones_output,
    convertor_in_out_states.rail_zones_input,
    convertor_in_out_states.rail_zones_state
)
def rail_zones_field_interface(selected_zones, min_values, max_values):
    return converter_utils.rail_zones_field_interface(
        selected_zones, min_values, max_values
    )


@app_convertor.callback(
    convertor_in_out_states.buttons_disabler_output,
    convertor_in_out_states.buttons_disabler_input,
    prevent_initial_call=True,
)
def buttons_disabler(
    signals_dropdown, methods_dropdown, zones_dropdown, zones_min_len,
    zones_max_len
):
    return converter_utils.buttons_disabler(
        signals_dropdown, methods_dropdown, zones_dropdown, zones_min_len,
        zones_max_len
    )


@app_convertor.callback(
    convertor_in_out_states.convertor_calculator_output,
    convertor_in_out_states.convertor_calculator_input,
    convertor_in_out_states.convertor_calculator_state,
    prevent_initial_call=True,
)
def calculate(
    n_clicks, filter_flag, abs_flag, signals_dropdown, methods_dropdown,
    zones_dropdown, zones_min_len, zones_max_len
):
    if n_clicks > 0:
        convertor_settings = pd.DataFrame(
            {
                "filter_flag": filter_flag,
                "abs_flag": abs_flag,
                "signals_dropdown": [
                    ",".join(signal) for signal in signals_dropdown
                ],
                "methods_dropdown": methods_dropdown,
                "zones_dropdown": zones_dropdown,
                "zones_min_len": zones_min_len,
                "zones_max_len": zones_max_len
            }
        )

        if not os.path.exists(CONVERTOR_OUTPUT_DIR):
            os.mkdir(join(PATH_TO_DATA, CONVERTOR_OUTPUT_DIR))
        convertor_settings.to_csv(
            join(PATH_TO_DATA, CONVERTOR_OUTPUT_DIR, "settings.csv")
        )
        return [
            [
                "Для получения информации о вычислениях перейдите "
                "на вкладку 'Статус' и нажмите кнопку 'Обновить'"
            ]
        ]
    return [[""]]


def remove_dir(data_to_remove):
    if os.path.exists(data_to_remove):
        shutil.rmtree(data_to_remove)


@app_convertor.callback(
    convertor_in_out_states.upload_output,
    convertor_in_out_states.upload_input,
    convertor_in_out_states.upload_state,
)
def upload_convertor_callback(list_of_contents, list_of_names, list_of_dates):
    remove_dir(join(PATH_TO_DATA, CONVERTOR_INPUT_DATA))
    remove_dir(join(PATH_TO_DATA, CONVERTOR_OUTPUT_DIR))

    if list_of_contents is not None:
        for content, name, date in zip(list_of_contents, list_of_names,
                                       list_of_dates):

            if "zip" in name:
                converter_utils.save_uploaded_data(
                    content, name, CONVERTOR_INPUT_DATA
                )

                return ["", CONVERTOR_INTERFACE]
            return ["", "Загрузите zip архив"]


####################
# Status callbacks #
####################


@app_status.callback(
    Output("calculation-status-table", "children"),
    Input('update-button', 'n_clicks')
)
def update_general_status(n_clicks):
    calculation_types = {
        "Конвертор": [
            os.path.join(PATH_TO_DATA, CONVERTOR_INPUT_DATA),
            os.path.join(PATH_TO_DATA, CONVERTOR_OUTPUT_DIR)
        ],
    }
    current_status = []
    for calculation_type, dirs in calculation_types.items():
        if os.path.exists(os.path.join(dirs[1], "alert.txt")):
            current_status.append(
                "В данных ошибка. Загрузите новые данные и попробуйте снова"
            )
        else:

            if (os.path.exists(os.path.join(
                    dirs[1],
                    "result.zip",
            )) or os.path.exists(os.path.join(
                    dirs[1],
                    "result.csv",
            )) or (os.path.exists(os.path.join(
                    dirs[1],
                    "separated_signals.xlsx",
            )) and not os.path.exists(os.path.join(
                    dirs[1],
                    "settings.csv",
            )))):
                current_status.append("Готов к загрузке")

            elif os.path.exists(dirs[1]):
                current_status.append("Вычисления")
            elif os.path.exists(dirs[0]):
                current_status.append("Ожидание вычислений")
            else:
                current_status.append("Нет данных для обработки")

    status = pd.DataFrame()
    status["Вычисление"] = pd.Series(calculation_types.keys())
    status["Статус"] = current_status

    children = html.Div(
        [
            dash_table.DataTable(
                status.to_dict('records'),
                [{
                    "name": i,
                    "id": i
                } for i in status.columns]
            )
        ]
    )

    return children


def generate_sender(path_to_data, input_dir, output_dir):
    print("Downloading data ...")
    sender = dcc.send_file(path_to_data)
    shutil.rmtree(output_dir)
    shutil.rmtree(input_dir)
    return sender


@app_status.callback(
    [Output("download-convertor-dataframe", "data")],
    Input(f'{CONVERTOR_OUTPUT_DIR}-button', 'n_clicks')
)
def update_convertor_result(n_clicks):
    if n_clicks > 0:
        input_dir = join(PATH_TO_DATA, CONVERTOR_INPUT_DATA)
        output_dir = join(PATH_TO_DATA, CONVERTOR_OUTPUT_DIR)
        path_to_data = join(output_dir, "result.csv")
        if os.path.exists(path_to_data):
            return [generate_sender(path_to_data, input_dir, output_dir)]


@server.route('/status/')
def calculation_status_dash_app():
    return app_status.index(debug=True)


@server.route("/convertor/")
def convertor_dash_app():
    return app_convertor.index(debug=True)


if __name__ == "__main__":
    app_convertor.run(
        host="0.0.0.0", port=os.getenv("DESTINATION_PORT", 80), debug=True
    )
