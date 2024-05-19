import dto, decomposer, new_generator
from dto import Source
from logger import logger
from dataclasses import dataclass
from typing import Dict
import pandas as pd
from datetime import datetime
from handlers import factory


@dataclass
class Processer:
    handlers_factory: factory.HandlersFactory
    metadata: pd.DataFrame
    constants: dto.Constants
    data_mapping: dict
    sources_settings: Dict[str, Source]

    def create_data_batch(self, billet_id):

        try:
            data_aggregator = decomposer.Decomposer(
                constants=self.constants,
                sources_settings=self.sources_settings)
            data_generator = new_generator.DataGenerator(
                constants=self.constants)

            pre_aggregated_sources, aggregated_sources = {}, {}
            bad_columns = []
            for source, filepath in self.data_mapping[billet_id].items():
                settings = self.sources_settings[source]
                data = pd.read_csv(filepath, sep=";", decimal=",", encoding=settings.encoding)
                source_handler = self.handlers_factory.get_handler(source)
                data["Stirng"] = ["a" for _ in range(data.shape[0])]
                source_handler.process_pipeline(data, settings)
                data = data.sort_values(by=settings.billet_column)
                data = data.drop(columns=settings.forbidden_columns)
                data = data[data[settings.billet_column].notna()]
                if not self.sources_settings[source].is_single:
                    bad_columns, data = self._check_columns_by_std(bad_columns, source, data)
                if settings.is_single:
                    data = data[data[settings.billet_column].str.contains(billet_id[:9])]
                for column in data.columns:
                    if column == settings.billet_column:
                        continue
                    data = self._convert_column_type(source, data, column, settings)
                    data = data.rename(columns={column: f"{source}_{column}"})
                data = data_generator.generate_secondary_features(
                    data, settings)
                pre_aggregated_sources[source] = data
                a = 1

            for source, _ in pre_aggregated_sources.items():
                if self.sources_settings[source].is_single:
                    aggregated_sources[source] = {list(self.sources_settings[source].segments.keys())[0]:
                                                  pre_aggregated_sources[source].to_dict('records')[0]}
                else:
                    data = self._interpolate_source(source, pre_aggregated_sources)
                    aggregated_data = data_aggregator.aggregate_segments(data, source)
                    aggregated_sources[source] = aggregated_data
            batch_aggregation = data_aggregator.aggregate_by_method(
                aggregated_sources, billet_id)
            for tg in range(len(batch_aggregation)):
                batch_aggregation[tg]["bad_columns"] = bad_columns
            logger.info(f"Успешно. Заготовка {billet_id}")
            return batch_aggregation
        except Exception:
            print(f"Ошибка. Заготовка {billet_id}")
            logger.error(f"Ошибка. Заготовка {billet_id}")
            logger.exception('')
            a = 1

    @staticmethod
    def _convert_column_type(source: str, data: pd.DataFrame, column: str, settings: Source) -> pd.DataFrame:
        if column in settings.convert_columns.keys():
            conv_type = settings.convert_columns[column]
            if conv_type == "datetime":
                data[column] = pd.to_datetime(data[column])
                data[column] = data[column].apply(
                    lambda x: (x - datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)).total_seconds())
            else:
                print(f"Source '{source}', column '{column}': Unknown convert type '{conv_type}'")
        return data

    def _interpolate_source(self, source: str, pre_data: dict):
        'Интерполяция'
        settings = self.sources_settings[source]
        data = pre_data[source]
        if settings.interpolation_type == 'by value':
            max_billet = max(data[settings.billet_column])
            data[settings.billet_column] = (
                    data[settings.billet_column]
                    * settings.interpolation / max_billet)
            for column in data.columns:
                if column == settings.billet_column:
                    continue
                data = data.rename(
                    columns={column: f"{column}_"
                                     f"[i_{settings.interpolation:.1f}]"})
        elif settings.interpolation_type == 'by source':
            max_billet = max(data[settings.billet_column])
            max_source_billet = max(
                pre_data[source][self.sources_settings[source].billet_column])
            data[settings.billet_column] = (
                    data[settings.billet_column]
                    * max_source_billet / max_billet)
            for column in data.columns:
                if column == settings.billet_column:
                    continue
                data = data.rename(
                    columns={column: f"{column}_[i_{max_source_billet:.1f}]"})
        else:
            for column in data.columns:
                if column == settings.billet_column:
                    continue
                data = data.rename(columns={column: f"{column}_[ni]"})
        return data

    @staticmethod
    def _check_columns_by_std(bad_columns: list, source: str, data: pd.DataFrame):
        lists = pd.read_csv("signal_lists.csv", delimiter=";")
        whitelist, blacklist = list(lists['whitelist']), list(lists['blacklist'])
        for colname, column_data in data.iteritems():
            if colname in blacklist:
                data = data.drop(columns=[colname])
            elif colname not in whitelist:
                try:
                    if column_data.std() == 0:
                        bad_columns.append(f"{source}_{colname}")
                except:
                    continue
        return bad_columns, data
