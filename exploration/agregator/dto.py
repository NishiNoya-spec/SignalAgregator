from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Segment:
    start_point: float
    end_point: float
    target_segments: List[str]


@dataclass
class SourceSettings:
    source: str
    type: str
    handler: str
    rolling_number: str
    encoding: str
    is_single: bool
    main_folder: str
    key_folder: str
    nested_folders: str
    filename_key: str
    interpolation_type: str
    interpolation: Optional[float]
    segments: Dict[str, Segment]
    forbidden_columns: List[str]
    filtration_methods: List[str]
    secondary_features: List[str]
    aggregation_methods: List[str]
    billet_column: str
    convert_columns: Dict[str, str]


@dataclass
class PipelineSetup:
    MIN_TIME: datetime    # ограничение файлов по минимальной дате
    MAX_TIME: datetime    # ограничение файлов по максимальной дате
    NUM_OF_CORES: int    # количество выделенных ядер ЦПУ
    MARK_FILTER: bool    # фильтровать или нет марки рельс
    MARK: List[str]    # марка рельса
    PATH_TO_MATERIALS: Optional[str]    # путь до вспомогательных материалов
    PATH_TO_METADATA: str    # пути до данных
    PATH_TO_RESULT: str    # путь до папки с результатом. Создавать самим
    METADATA_BILLET_ID: str    # имя столбца с ИД заготовки


@dataclass
class Materials:
    PATHS: dict


@dataclass
class CutPoint:
    L_DELTA_MIN: float
    L_DELTA_MAX: float
    L_WIN: int
    R_DELTA_MIN: float
    R_DELTA_MAX: float
    R_WIN: int
