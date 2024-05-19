import os
import re
from datetime import datetime
from glob import glob
from multiprocessing.pool import Pool
from typing import Dict, Tuple

import dto
import pandas as pd
from dto import SourceSettings
from logger import MainLogger


class MappingCreator:

    def __init__(self, setup: dto.PipelineSetup, logger: MainLogger):
        self.setup = setup
        self.logger = logger

    def get_metadata(self):
        billets_info = list()
        metadata_files = glob(self.setup.PATH_TO_METADATA)

        if not metadata_files:
            error_msg = "Не найдено файлов Metadata. Процесс остановлен"
            self.logger.flog(error_msg, is_print=False, log_type="error")
            raise FileNotFoundError(error_msg)

        for metadata_path in glob(self.setup.PATH_TO_METADATA):
            metadata = pd.read_excel(metadata_path, engine='openpyxl')
            metadata[self.setup.METADATA_BILLET_ID] = \
                "Л2" + \
                metadata["Плавка"].str.split("-", expand=True).iloc[:, -1] + \
                metadata["Руч"].astype(str) + "0" + \
                metadata["Заг"].astype(str) + "_" + \
                metadata["Вр.реза"].dt.year.astype(str)

            if self.setup.MARK_FILTER is True:
                metadata = metadata[metadata["Марка"].isin(self.setup.MARK)]
            billets_info.append(
                metadata[["Вр.проката", "BilletId", "Марка"]].dropna()
            )
        billets_metadata = pd.concat(billets_info)
        self.logger.flog(
            f"Обнаружено {len(metadata_files)} файл(а) Metadata, "
            f"в которых содержатся данные {billets_metadata.shape[0]} "
            f"плавок",
            is_print=False,
            log_type="info"
        )
        return pd.concat(billets_info).drop_duplicates(subset=["BilletId"])

    def create_mapping(
        self, metadata: pd.DataFrame, settings: Dict[str, SourceSettings]
    ) -> Tuple[dict, pd.DataFrame]:

        sources = {}
        for source_name, source in settings.items():
            filepaths = []

            files_folders = self._get_key_folders(
                main_folder=source.main_folder, key=source.key_folder
            )
            for folder in files_folders:
                filepaths.extend(
                    glob(
                        os.path.join(
                            folder, f"{source.nested_folders}", "*csv"
                        )
                    )
                )

            filtered_filepaths = []
            for filepath in filepaths:
                if source.filename_key in filepath:
                    filtered_filepaths.append(filepath)

            if not filtered_filepaths:
                error_msg = f"Не найдено файлов источника {source_name}. " \
                            f"Процесс остановлен"
                self.logger.flog(error_msg, is_print=False, log_type="error")
                raise FileNotFoundError(error_msg)
            else:
                self.logger.flog(
                    f"Обнаружено {len(filtered_filepaths)} "
                    f"файлов источника {source_name}",
                    is_print=False,
                    log_type="info"
                )

            if source.is_single:
                inputs = [
                    (filepath, metadata, source)
                    for filepath in filtered_filepaths
                ]
                with Pool(self.setup.NUM_OF_CORES) as pool:
                    billets = pool.starmap(self._get_single_billet_id, inputs)
                sources[source_name] = dict(
                    [billet for pack in billets for billet in pack]
                )
            else:
                with Pool(self.setup.NUM_OF_CORES) as pool:
                    billets = pool.map(self._get_billet_id, filtered_filepaths)
                sources[source_name] = dict(
                    [billet for billet in billets[:] if billet is not None]
                )

        by_billets = dict()

        not_use_billets = {key: [] for key in sources.keys()}
        not_use_billets["BilletId"] = []
        not_use_billets["Time"] = []
        for billet_id in metadata[self.setup.METADATA_BILLET_ID].to_list():
            if all([billet_id in billet_ids
                    for _, billet_ids in sources.items()]):
                by_billets[billet_id] = dict(
                    [
                        (source, sources[source][billet_id])
                        for source in sources
                    ]
                )
            else:
                not_use_billets["BilletId"].append(billet_id)
                not_use_billets["Time"].append(
                    metadata[metadata[self.setup.METADATA_BILLET_ID] ==
                             billet_id]["Вр.проката"].iloc[0]
                )
                for source, billet_ids in sources.items():
                    is_in_billet_ids = int(billet_id in billet_ids)
                    not_use_billets[source].append(is_in_billet_ids)
        self.logger.flog(
            f"Создание маппинга завершено. Всего обнаружено "
            f"{len(by_billets)} уникальных заготовок",
            is_print=False,
            log_type="info"
        )

        not_use_billets = pd.DataFrame(not_use_billets)
        source_columns = list(sources.keys())
        not_use_billets["fullness"] = not_use_billets[source_columns].sum(
            axis=1
        ).astype(str) + "/" + str(len(source_columns))
        not_use_billets = not_use_billets[
            ["Time", self.setup.METADATA_BILLET_ID] + source_columns
            + ["fullness"]]
        return by_billets, not_use_billets

    def _get_billet_id(self, filepath: str) -> (str, str):
        file_name, file_ext = os.path.basename(filepath).split('.')
        if "_L" in file_name or "_T" in file_name:
            if re.findall(r'Л\d{8}', file_name):
                billet_id = re.findall(r'Л\d{8}', file_name)[0]
                return self._get_billet_from_period(
                    file_name[:14], billet_id, filepath
                )

            elif re.findall(r'\d{5}X\d{3}', file_name):
                billet_id = re.findall(r'\d{5}X\d{3}', file_name)
                billet_id = "Л" + billet_id[0].replace("X", "")
                return self._get_billet_from_period(
                    file_name[:14], billet_id, filepath
                )

    def _get_billet_from_period(self, date: str, billet_id: str, filepath):
        billet_time = datetime.strptime(date, "%Y%m%d%H%M%S")
        if self._is_billet_in_period(billet_time):
            return f"{billet_id}_{billet_time.year}", filepath

    def _is_billet_in_period(self, billet_time) -> bool:
        return self.setup.MIN_TIME <= billet_time <= self.setup.MAX_TIME

    def _get_key_folders(self, main_folder: str, key: str) -> list:
        folders = []
        for (_, directories, _) in os.walk(main_folder):
            if key in directories:
                return [os.path.join(main_folder, f"{key}")]
            elif not directories:
                return []
            else:
                for directory in directories:
                    folders.extend(
                        self._get_key_folders(
                            os.path.join(main_folder, directory), key
                        )
                    )
                return folders

    def _get_single_billet_id(
        self, filepath: str, metadata: pd.DataFrame, source: SourceSettings
    ):
        filepath_mapping = []
        data_csv = pd.read_csv(
            filepath, delimiter=";", encoding=source.encoding
        )
        for file_billet in data_csv[source.billet_column]:
            billet_data = metadata[metadata["BilletId"].str.contains(
                file_billet[:9]
            )]
            if len(billet_data) == 0:
                continue
            if self._is_billet_in_period(billet_data['Вр.проката'].iloc[0]):
                filepath_mapping.append(
                    (billet_data["BilletId"].iloc[0], filepath)
                )
        return filepath_mapping
