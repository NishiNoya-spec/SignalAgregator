import math
import os
import pickle
from glob import glob
from multiprocessing import pool
from typing import Dict, List, Tuple

import constants
import data_mapping
import pandas as pd
import processer
from components.features_generator import FeaturesGenerator
from components.filters import Filters
from components.secondary_functions import SecondaryFunctions
from components.segments_aggregator import (
    AggregatedSegment,
    SegmentsAggregator,
)
from dto import Materials, PipelineSetup, SourceSettings
from handlers.factory import HandlersFactory
from logger import MainLogger
from matcher import Matcher
from source_parser import parse_settings
from tqdm import tqdm


class MainPipeline:

    def __init__(self, setup: PipelineSetup):
        self.setup = setup

    def run_pipeline(self):
        """Основной пайплайн выполнения агрегации"""
        # Создаем пути для сохранения файлов
        self._create_save_paths()

        # Объект Логгера
        logger = MainLogger(
            os.path.join(self.setup.PATH_TO_RESULT, constants.LOGS_FOLDER)
        )
        logger.flog("Агрегация запущена", log_type="info")

        # Считываем и парсим настройки агрегации
        settings = pd.read_excel(
            "settings.xlsx", index_col=[0], header=None
        ).iloc[:, :-1]
        sources_settings = parse_settings(settings)
        self._add_target_paths(sources_settings)
        logger.flog("Настройки источников обработаны", log_type="info")

        # парсинг метаинформации
        creator = data_mapping.MappingCreator(setup=self.setup, logger=logger)
        metadata = creator.get_metadata()
        logger.flog("Метадата обработана", log_type="info")

        # Создание мапинга файлов
        mapping, report = creator.create_mapping(
            metadata=metadata, settings=sources_settings
        )
        logger.flog(
            f"Мапинг создан. Всего {len(mapping)} уникальных заготовок",
            log_type="info"
        )

        # Компоненты запуска процесса обработки файлов
        materials = Materials(
            {
                os.path.splitext(os.path.basename(path))[0]: path
                for path in glob(self.setup.PATH_TO_MATERIALS)
            } if self.setup.PATH_TO_MATERIALS else None
        )
        features_generator = FeaturesGenerator()
        secondary_functions = SecondaryFunctions()
        segments_aggregator = SegmentsAggregator()
        filters = Filters()

        # Фабрика сборки кастомных пайплайнов
        handlers_factory = HandlersFactory(
            features_generator=features_generator,
            filters=filters,
            secondary_functions=secondary_functions,
            segments_aggregator=segments_aggregator,
            materials=materials
        )

        # Класс для сопоставления данных фичей с таргетом
        matcher = Matcher(sources_settings)

        # Основной обработчик
        main_processer = processer.Processer(
            setup=self.setup,
            metadata=metadata,
            data_mapping=mapping,
            sources_settings=sources_settings,
            handlers_factory=handlers_factory,
            matcher=matcher
        )

        # Оркестратор
        multiproc_queue = []
        for n in range(math.ceil(len(mapping) / self.setup.NUM_OF_CORES)):
            multiproc_queue.append(
                list(mapping.keys())[(self.setup.NUM_OF_CORES
                                      * n):(self.setup.NUM_OF_CORES * (n + 1))]
            )
        logger.flog(
            f"Заготовки распределены по {self.setup.NUM_OF_CORES} ядрам. "
            f"Всего {len(multiproc_queue)} итераций",
            log_type="info"
        )

        # Сохранение метаинформации агрегатора
        processing_dates = metadata["Вр.проката"].dt.date
        self._save_settings(
            {
                "FIRST_DATE": str(processing_dates.min()),
                "LAST_DATE": str(processing_dates.max())
            }
        )
        self._save_report(report)

        logger.flog("Запуск аггрегации.", log_type="info")
        for cut_num in tqdm(range(len(multiproc_queue)),
                            bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}'):
            cut = multiproc_queue[cut_num]
            # Debug
            # deb = main_processer.create_data_batch(cut[1])

            with pool.Pool(self.setup.NUM_OF_CORES) as p:
                batch = p.map(main_processer.create_data_batch, cut)

            self._save_batch(cut, batch)

    def _create_save_paths(self):
        """Создаем папки для результатов аггрегации"""
        # Корневая папка эксперимента
        os.makedirs(self.setup.PATH_TO_RESULT, exist_ok=True)

        # Папка с настройками и логами
        os.makedirs(
            os.path.join(self.setup.PATH_TO_RESULT, constants.LOGS_FOLDER),
            exist_ok=True
        )

        # Папка с изначальными данными
        os.makedirs(
            os.path.join(
                self.setup.PATH_TO_RESULT, constants.INITIAL_DATA_FOLDER
            ),
            exist_ok=True
        )

        # Папки с сагреггированными данными
        aggregated_data_folder_path = os.path.join(
            self.setup.PATH_TO_RESULT, constants.PREPARED_DATA_FOLDER
        )
        os.makedirs(aggregated_data_folder_path, exist_ok=True)

    def _add_target_paths(self, source_settings: Dict[str, SourceSettings]):
        """Создаем папки для таргетов"""
        aggregated_data_folder_path = os.path.join(
            self.setup.PATH_TO_RESULT, constants.PREPARED_DATA_FOLDER
        )
        for source in source_settings.values():
            if source.type == 'target':
                for target_segment in source.segments.keys():
                    os.makedirs(
                        os.path.join(
                            aggregated_data_folder_path, target_segment
                        ),
                        exist_ok=True
                    )

    def _save_settings(self, other_setups: Dict[str, str]):
        """Сохраняем общие настройки аггрегации"""
        path_to_settings = os.path.join(
            self.setup.PATH_TO_RESULT, constants.LOGS_FOLDER, 'setup.txt'
        )
        with open(path_to_settings, "w") as handle:
            handle.write(
                f"NUM_OF_CORES = {self.setup.NUM_OF_CORES}\n"
                f"MIN_TIME = {str(self.setup.MIN_TIME)}\n"
                f"MAX_TIME = {str(self.setup.MAX_TIME)}\n"
                f"MARK_FILTER = {self.setup.MARK_FILTER}\n"
                f"MARK = {self.setup.MARK}\n"
                f"METADATA_BILLET_ID = {self.setup.METADATA_BILLET_ID}\n"
            )
            for key, value in other_setups.items():
                handle.write(f"{key} = {value} \n")

    def _save_report(self, report: pd.DataFrame):
        """Сохраняем результаты маппинга по всем заготовкам"""
        report_dir = os.path.join(
            self.setup.PATH_TO_RESULT, constants.LOGS_FOLDER, "report.csv"
        )
        report.to_csv(report_dir, sep=";", decimal=",")

    def _save_batch(
        self, cut: List[str], batch: List[Tuple[Dict[str, AggregatedSegment],
                                                Dict[str, Dict[str, float]]]]
    ):
        """Сохраняем результаты аггрегации"""
        for billet_data, billet_id in zip(batch, cut):
            initial_path = os.path.join(
                self.setup.PATH_TO_RESULT, constants.INITIAL_DATA_FOLDER
            )
            with open(os.path.join(initial_path, f"{billet_id}.pkl"),
                      "wb") as handle:
                pickle.dump(billet_data[0], handle)
            for target_segment, values in billet_data[1].items():
                prepared_path = os.path.join(
                    self.setup.PATH_TO_RESULT, constants.PREPARED_DATA_FOLDER,
                    target_segment
                )
                with open(os.path.join(prepared_path, f"{billet_id}.pkl"),
                          "wb") as handle:
                    pickle.dump(values, handle)

    @staticmethod
    def get_time_period_in_str(time_period: pd.Series):
        processing_dates = time_period.dt.date
        return str(processing_dates.min()) + "_" + str(processing_dates.max())
