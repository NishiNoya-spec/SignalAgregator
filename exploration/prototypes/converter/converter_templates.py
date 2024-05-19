from glob import glob

from dash import dcc, html

from exploration.prototypes.converter.core.converter import (
    AGG_METHODS,
    RAILE_ZONE,
    pd,
)

top_style = {
    "width": "30%",
    'textAlign': 'center',
    "display": "flex",
    "align-items": "center",
    "flex-direction": "row",
    "justify-content": "space-around"
}
upload_style = {
    'width': '80%',
    'height': '4em',
    'lineHeight': '2em',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '1em',
    'textAlign': 'center',
    'display': 'inline-block',
}
main_style = {
    "width": "100%",
    'textAlign': 'center',
    "display": "flex",
    "justify-content": "center",
    "align-items": "center",
    "flex-direction": "column"
}

doc_link = "http://tpqc-docs-zsmk-9684-dev.apps.ocpd." \
           "sib.evraz.com/algorithms/convertor/index.html"

TOP_ROW = html.Div(
    [
        dcc.Link(href=doc_link, refresh=True, children="Документация"),
        dcc.Link(href="/", refresh=True, children="Конвертор"),
        dcc.Link(href="/status/", refresh=True, children="Статус")
    ],
    style=top_style
)
UPLOAD_INTERFACE = dcc.Input(
    id="input-data",
    placeholder="Укажите путь",
)

CONVERTOR_PAGE_TEMPLATE = html.Div(
    [
        TOP_ROW,
        html.H2("Конвертор в показатели"),
        html.P(
            "Укажите путь до папки с данными. "
            "Данные внутри папки должны быть в .csv формате"
        ), UPLOAD_INTERFACE,
        html.Button('Загрузить', id='upload-button', n_clicks=0),
        dcc.Loading(
            id="loading-uploading",
            children=[html.Div(id="loading-output-uploading")],
            type="default"
        ),
        html.Div(id="convertor-interface"),
        html.Div(id="empty-redirect"),
        dcc.Interval(
            id='interval-download-component', interval=10000, n_intervals=0
        ),
        dcc.Download(id="download-dataframe")
    ],
    id="aggregator_settings",
)

CONVERTOR_INTERFACE = [
    html.H4(
        "Выберите параметры конвертации, запустите "
        "вычисления, и ожидайте автоматической загрузки результата"
    ),
    html.Div(
        [
            html.Div("Сигналы", style={
                "text-align": "center",
                "width": "25%"
            }),
            html.Div(
                "Метод агрегации",
                style={
                    "text-align": "center",
                    "width": "10%"
                }
            ),
            html.Div(
                "Зона рельса", style={
                    "text-align": "center",
                    "width": "20%"
                }
            ),
            html.Div("от, м", style={
                "text-align": "center",
                "width": "3%"
            }),
            html.Div("до, м", style={
                "text-align": "center",
                "width": "3%"
            }),
            html.Div("Фильтр", style={
                "text-align": "center",
                "width": "10%"
            }),
            html.Div(
                "Абсолютные значения",
                style={
                    "text-align": "center",
                    "width": "10%"
                }
            ),
            html.Button(
                "Новый конвертор",
                id='add-row',
                n_clicks=0,
                style={
                    "text-align": "center",
                    "width": "19%"
                },
                disabled=False
            ),
        ],
        style={
            "display": "flex",
            "flex-direction": "row"
        }
    ),
    html.Div(id="rows-container", children=[]),
    html.Button(
        'Запуск вычислений',
        id='calculations-runner',
        n_clicks=0,
        disabled=True
    ),
    html.Div(id="calculations-info", children=""),
    dcc.Loading(
        id="loading-1",
        children=html.Div(id="loading-output-1"),
        type="default"
    ),
]

STATUS_PAGE = html.Div(
    [
        TOP_ROW,
        html.H2("Вычисления"),
        html.Button('Обновить', id='update-button', n_clicks=0),
        dcc.Download(id="download-convertor-dataframe"),
        html.Div([], id="calculation-status-table"),
        html.Button(
            'Загрузить Конвертор', id='convertor-result-button', n_clicks=0
        ),
    ]
)


class ConverterInterfaceElementsGenerator:

    def __init__(self):
        self.folder_path = ''
        self.data_paths = []
        self.signal_names_for_aggregation = []

    def set_data_paths_and_signal_names_for_aggregation(self, data_path):
        self.folder_path = data_path.split("*")[0]
        self.data_paths = glob(data_path)

        signal_names_for_aggregation = list(
            pd.read_csv(self.data_paths[0], sep=";",
                        decimal=",").columns.to_series().reset_index(drop=True
                                                                     )[1:]
        )

        if "BilletPoint" in signal_names_for_aggregation:
            signal_names_for_aggregation.remove("BilletPoint")
        elif "billet_points" in signal_names_for_aggregation:
            signal_names_for_aggregation.remove("billet_points")

        self.signal_names_for_aggregation = signal_names_for_aggregation

    def generate_dropdown(
        self, dropdown_id: str, index: int, options: list, multi=False
    ):
        return dcc.Dropdown(
            id={
                'type': dropdown_id,
                'index': index
            },
            options=options,
            multi=multi,
            style={"width": "100%"}
        )

    def generate_check_list(
        self,
        check_list_id: str,
        index: int,
        options: list,
        value: list,
        inline=True
    ):
        return dcc.Checklist(
            options,
            value,
            id={
                'type': check_list_id,
                'index': index
            },
            inline=inline
        )

    def generate_input_number(self, input_number_id: str, index: int):
        return html.Div(
            [
                dcc.Input(
                    id={
                        'type': input_number_id,
                        'index': index
                    },
                    style={"width": "100%"},
                    type="number"
                )
            ],
            style={"width": "15%"}
        )

    def generate_interface_row(self, index):
        return html.Div(
            [
                html.Div(
                    self.generate_dropdown(
                        'signals-dropdown-id',
                        index,
                        self.signal_names_for_aggregation,
                        multi=True
                    ),
                    style={
                        "text-align": "center",
                        "width": "25%"
                    }
                ),
                html.Div(
                    self.generate_dropdown(
                        'methods-dropdown-id', index, [
                            {
                                'label': label,
                                'value': value
                            } for label, value in AGG_METHODS.items()
                        ]
                    ),
                    style={
                        "text-align": "center",
                        "width": "10%"
                    }
                ),
                html.Div(
                    [
                        self.generate_dropdown(
                            'zones-dropdown-id', index, [
                                {
                                    'label': label,
                                    'value': value
                                } for label, value in RAILE_ZONE.items()
                            ]
                        ),
                        self.generate_input_number('zones-min-len-id', index),
                        self.generate_input_number('zones-max-len-id', index),
                    ],
                    style={
                        "text-align": "center",
                        "width": "26%",
                        "display": "flex",
                        "flex-direction": "row"
                    }
                ),
                html.Div(
                    self.generate_check_list('filter-flag', index, [""], []),
                    style={
                        "text-align": "center",
                        "width": "10%"
                    }
                ),
                html.Div(
                    self.generate_check_list('abs-flag', index, [""], []),
                    style={
                        "text-align": "center",
                        "width": "10%"
                    }
                )
            ],
            style={
                "display": "flex",
                "flex-direction": "row"
            }
        )
