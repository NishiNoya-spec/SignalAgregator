import ast
from typing import Dict

import pandas as pd
from dto import Segment, SourceSettings


def parse_settings(settings: pd.DataFrame) -> Dict[str, SourceSettings]:
    sources = {}
    for _, source_settings in settings.iteritems():
        source_settings = source_settings.fillna("")

        # Парсим сегменты
        segments = {}
        for segment_data in str(source_settings["segments"]).split("\n"):
            segment_id, start, end, target_ids = eval(segment_data)
            segments[f"{source_settings['source']}_{segment_id}"] = Segment(
                start_point=start, end_point=end, target_segments=target_ids
            )
        # Парсим специальные колонки
        convert_columns = {}
        if source_settings["convert_columns"] != '':
            for convert_col in str(source_settings["convert_columns"]
                                   ).split("\n"):
                colname, convert_type = eval(convert_col)
                convert_columns[colname] = convert_type

        # Парсим остальные настройки источника
        if source_settings['interpolation_type'] == 'by value':
            interp = float(source_settings['interpolation'])
        elif source_settings['interpolation_type'] == 'by source':
            interp = source_settings['interpolation']
        elif source_settings['interpolation_type'] == "":
            interp = None
        else:
            raise NameError(
                f"Unknown interpolation type for source "
                f"{source_settings['source']}"
            )

        sources[source_settings['source']] = SourceSettings(
            source=source_settings['source'],
            type=source_settings['type'],
            handler=source_settings['handler'],
            rolling_number=str(source_settings['rolling_number']),
            encoding=source_settings['encoding'],
            is_single=source_settings['is_single'],
            main_folder=source_settings['main_folder'],
            key_folder=source_settings['key_folder'],
            nested_folders=source_settings['nested_folders'],
            filename_key=source_settings['filename_key'],
            interpolation_type=source_settings['interpolation_type'],
            billet_column=source_settings['billet_column'],
            interpolation=interp,
            segments=segments,
            forbidden_columns=ast.literal_eval(
                source_settings['forbidden_columns']
            ),
            filtration_methods=ast.literal_eval(
                source_settings['filtration_methods']
            ),
            secondary_features=ast.literal_eval(
                source_settings['secondary_features']
            ),
            aggregation_methods=ast.literal_eval(
                source_settings['agg_methods']
            ),
            convert_columns=convert_columns
        )

    return sources
