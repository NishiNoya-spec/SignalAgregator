from dash import ALL
from dash.dependencies import Input, Output, State


class ConverterInputOutputStates:

    def __init__(self):
        self.dropdown_output = Output('rows-container', 'children')
        self.dropdown_input = Input('add-row', 'n_clicks')
        self.dropdown_state = State('rows-container', 'children')

        self.rail_zones_output = [
            Output({
                'type': 'zones-min-len-id',
                'index': ALL
            }, 'value'),
            Output({
                'type': 'zones-max-len-id',
                'index': ALL
            }, 'value'),
            Output({
                'type': 'zones-min-len-id',
                'index': ALL
            }, 'disabled'),
            Output({
                'type': 'zones-max-len-id',
                'index': ALL
            }, 'disabled')
        ]
        self.rail_zones_input = Input(
            {
                'type': 'zones-dropdown-id',
                'index': ALL
            }, 'value'
        )
        self.rail_zones_state = [
            State({
                'type': 'zones-min-len-id',
                'index': ALL
            }, 'value'),
            State({
                'type': 'zones-max-len-id',
                'index': ALL
            }, 'value')
        ]

        self.buttons_disabler_output = [
            Output('calculations-runner', 'disabled'),
            Output('add-row', 'disabled')
        ]
        self.buttons_disabler_input = [
            Input({
                'type': 'signals-dropdown-id',
                'index': ALL
            }, 'value'),
            Input({
                'type': 'methods-dropdown-id',
                'index': ALL
            }, 'value'),
            Input({
                'type': 'zones-dropdown-id',
                'index': ALL
            }, 'value'),
            Input({
                'type': 'zones-min-len-id',
                'index': ALL
            }, 'value'),
            Input({
                'type': 'zones-max-len-id',
                'index': ALL
            }, 'value')
        ]

        self.upload_output = [
            Output('loading-output-uploading', 'children'),
            Output('convertor-interface', 'children')
        ]
        self.upload_input = [Input('input-data', 'contents')]
        self.upload_state = [
            State('input-data', 'filename'),
            State('input-data', 'last_modified')
        ]

        self.convertor_calculator_output = [
            Output("loading-output-1", "children")
        ]
        self.convertor_calculator_input = Input(
            'calculations-runner', 'n_clicks'
        )
        self.convertor_calculator_state = [
            State({
                'type': 'filter-flag',
                'index': ALL
            }, 'value'),
            State({
                'type': 'abs-flag',
                'index': ALL
            }, 'value'),
            State({
                'type': 'signals-dropdown-id',
                'index': ALL
            }, 'value'),
            State({
                'type': 'methods-dropdown-id',
                'index': ALL
            }, 'value'),
            State({
                'type': 'zones-dropdown-id',
                'index': ALL
            }, 'value'),
            State({
                'type': 'zones-min-len-id',
                'index': ALL
            }, 'value'),
            State({
                'type': 'zones-max-len-id',
                'index': ALL
            }, 'value')
        ]

        self.update_result_output = Output("download-dataframe", "data")
        self.update_result_input = Input(
            'interval-download-component',
            'n_intervals',
        )

        self.update_status_output = Output("calculations-info", "children")
        self.update_status_input = Input('interval-component', 'n_intervals')


class CutterInputOutputStates:

    def __init__(self):
        pass
