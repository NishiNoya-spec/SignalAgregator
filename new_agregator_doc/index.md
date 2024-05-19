# Описание пайплайна агрегирования сигналов

Список версий предустановленных библиотек:

- numpy==1.21.6
- pandas==1.3.5
- Pympler==1.0.1
- openpyxl==3.1.2
- tqdm==4.65.0

Дерево проекта для сборки агрегатора:

    └── agregator
        ├── components
        |    ├── __init__.py
        |    ├── features_generator.py
        |    ├── filters.py
        |    ├── secondary_functions.py
        |    └── segments_aggregator.py
        |    
        ├── handlers
        |    ├── __init__.py
        |    ├── BASE_handler.py
        |    ├── BASE_WITH_POINTS.py
        |    ├── factory.py
        |    ├── interfaces.py
        |    ├── WBF_PIRO_handler.py
        |    └── WBF_SINGLE_handler.py
        |    
        ├── run
        |    ├── materials
        |    |   ├── max_data.json
        |    |   ├── min_data.json
        |    |   ├── signal_lists.json
        |    |   ├── wbf_piro_cutter_settings.json
        |    |   └── workcenter.json
        |    |
        |    ├── run.py
        |    └── settings.xlsx
        |
        ├── __init__.py
        ├── constants.py
        ├── data_mapping.py
        ├── decomposer.py
        ├── dto.py
        ├── logger.py
        ├── matcher.py
        ├── naming.py
        ├── pipeline.py
        ├── processor.py
        ├── source_data.py
        └── source_parser.py

Запуск пайплайна агрегирования сигналов начинается в модуле `run`.

## Пример настройки пайплайна

    setup = dto.PipelineSetup(
        PATH_TO_RESULT=r"\\ZSMK-9684-001\Data\DS\test_new_prep",
        NUM_OF_CORES=5,
        MIN_TIME=datetime(year=2023, month=11, day=10, hour=0, minute=0, second=0),
        MAX_TIME=datetime(year=2023, month=11, day=12, hour=0, minute=0, second=0),
        MARK_FILTER=True,
        MARK=['Э76ХФ', 'Э90ХАФ', '76ХФ', '90ХАФ'],
        PATH_TO_METADATA=os.path.join(
            r'\\ZSMK-9684-001\Data\DS', "metadata/debug/*xlsx"
        ),
        PATH_TO_MATERIALS=r"agregator\run\materials\*",
        METADATA_BILLET_ID="BilletId"
    )

Настройки пайплайна, определенные в структурах данных модуля `dto` как `PipelineSetup`.

- PATH_TO_RESULT: Путь к папке, в которую будут сохранены результаты пайплайна.
- NUM_OF_CORES: Количество ядер процессора, используемых для параллельной обработки.
- MIN_TIME: Ограничение данных по минимальной дате.
- MAX_TIME: Ограничение данных по максимальной дате.
- MARK_FILTER: Флаг для фильтрации по маркам рельс.
- MARK: Список марок рельс для фильтрации.
- PATH_TO_METADATA: Пути до данных.
- PATH_TO_MATERIALS: Путь до вспомогательных материалов.
- METADATA_BILLET_ID: Имя столбца с идентификатором заготовки.


## Модуль `dto` структур данных

#### Структура данных 'Segment'
Структура данных 'Segment' необходима для описания объектов-сегментов данных.

    @dataclass
    class Segment:
        start_point: float # Начальная точка
        end_point: float # Конечная точка 
        target_segments: List[str] # Имена сегментов target, к которым будет сопоставлен данный сегмент

Пример сегментов по времени:

    Segment(start_point='2024-03-07 23:59:28.907000', end_point='2024-03-08 00:01:07.607000', target_segments=['LNK100_1'])

Пример сегментов по длине заготовки:

    Segment(start_point=0, end_point=3, target_segments=["LNK100_1"])


#### Структура данных 'SourceSettings'
Структура данных 'SourceSettings' необходима для описания настроек агрегации.

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

Структура `SourceSettings` содержит cледующие настроечные параметры:

- `source`: Имя источника данных, фичи (LNK100; U0; BD1; WBF_PIRO).
- `type`: Тип источника данных, фичи (target; feature).
- `handler`: Обработчик данных (BASE; WBFPIRO; WBFSINGLE).
- `rolling_number`: Номер прокатки.
- `encoding`: Кодировка данных (UTF-8; ANSI).
- `is_single`: Флаг, указывающий, что параметры сигнала имеют одно значение или более.
- `main_folder`: Корневая папка, с которой стартует рекурсивный поиск.
- `key_folder`: Ключевая папка источника данных.
- `nested_folders`: Вложенная папка внутри key_folder.
- `filename_key`: Ключ имени файла.
- `interpolation_type`: Тип интерполяции (`by_value`; `by_source`).
- `interpolation`: Значение интерполяции (необязательно).
- `segments`: Сегменты на которые разбиватеся заготовка.
- `forbidden_columns`: Имена колонок которые можно выкинуть из источника.
- `filtration_methods`: Методы фильтрации.
- `secondary_features`: вспомогательные функции для обработки и анализа данных.
- `aggregation_methods`: Методы аггрегации данных внутри сегмента.
- `billet_column`: Имя колонки с Billet points.
- `convert_columns`: Словарь для преобразования столбцов.


#### Структура данных 'PipelineSetup'
Структура данных 'PipelineSetup' необходима для основных настроек пайплайна.

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


#### Структура данных 'Materials'
Структура данных 'Materials' определяет словарь, который содержит пути до вспомогательных данных.

    @dataclass
    class Materials:
        PATHS: dict

Пример:

    materials = {
        'max_data': 'agregator/run/materials/max_data.json',
        'min_data': 'agregator/run/materials/min_data.json',
        'signal_lists': 'agregator/run/materials/signal_lists.csv',
        'wbf_piro_cutter_settings': 'agregator/run/materials/wbf_piro_cutter_settings.json',
        'workcenters': 'agregator/run/materials/workcenters.json'
    }


#### Структура данных 'CutPoint'
Структура данных 'CutPoint' определяет параметры для выделения точек разделения сигналов, это необходимо 
для выделения наиболее подходящих сегментов сигналов с пирометров ПШБ.

Точки разделения задаются в формате левого окна `L_WIN`, правого окна `R_WIN` и допустимого изменения сигнала
`L_DELTA_MIN`, `L_DELTA_MAX`, `R_DELTA_MIN`, `R_DELTA_MAX`:

    @dataclass
    class CutPoint:
        L_DELTA_MIN: float # Минимальное изменение слева.
        L_DELTA_MAX: float # Максимальное изменение слева.
        L_WIN: int # Размер окна слева.
        R_DELTA_MIN: float # Минимальное изменение справа.
        R_DELTA_MAX: float # Максимальное изменение справа.
        R_WIN: int # Размер окна справа.


# MainPipeline Class. Пример запуска основного пайплайна   
Для запуска пайплайна агрегации сигналов необходимо создать объект `pipeline` класса `MainPipeline` и передать в него 
раннее созданный объект `setup` с настройками пайплайна. Далее необходимо вызвать метод `run_pipeline` объекта `pipeline`:

    def run_aggregation():
        pipeline = MainPipeline(setup)
        pipeline.run_pipeline()

    if __name__ == "__main__":
        run_aggregation()

Ниже приведен пример имплементации метода `run_pipeline` класса `MainPipeline`:

    class MainPipeline:

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


Далее будет приведен пошаговый разбор работы пайплайна.

---

# 1. Создание путей для сохранения файлов.

Вызывается метод `_create_save_paths()`, который создает необходимые папки для сохранения результатов агрегации.

    def _create_save_paths(self):
        """
        Создает папки для сохранения результатов агрегации.
        """

        # Создаем корневую папку результатов агрегации (при необходимости)
        os.makedirs(self.setup.PATH_TO_RESULT, exist_ok=True)
    
        # Создаем папку для логов (при необходимости)
        os.makedirs(
            os.path.join(self.setup.PATH_TO_RESULT, constants.LOGS_FOLDER),
            exist_ok=True
        )
    
        # Создаем папку для изначальных данных (при необходимости)
        os.makedirs(
            os.path.join(
                self.setup.PATH_TO_RESULT, constants.INITIAL_DATA_FOLDER
            ),
            exist_ok=True
        )
    
        # Создаем папку для сагрегированных данных (при необходимости)
        aggregated_data_folder_path = os.path.join(
            self.setup.PATH_TO_RESULT, constants.PREPARED_DATA_FOLDER
        )
        os.makedirs(aggregated_data_folder_path, exist_ok=True)

Эта функция создает папки для сохранения результатов агрегации данных.
Каждая папка создается в соответствии с заданным путем к корневой папке результатов.
Путь к корневой папке результатов указывается при инициализации объекта MainPipeline и хранится в атрибуте PATH_TO_RESULT объекта setup.
При этом используются константы, определенные в модуле `constants`, для имени каждой папки:

- `constants.LOGS_FOLDER`: Папка для логов.
- `constants.INITIAL_DATA_FOLDER`: Папка для изначальных данных.
- `constants.PREPARED_DATA_FOLDER`: Папка для агрегированных данных.

---

# 2. Создание объекта Логгера.

Инициализируется объект `logger` класса `MainLogger`, который используется для логирования действий во время выполнения пайплайна.

    # Объект Логгера
    logger = MainLogger(
        os.path.join(self.setup.PATH_TO_RESULT, constants.LOGS_FOLDER)
    )

### MainLogger Class

Класс представляет объект для логирования операций во время выполнения агрегации данных.

`__init__(self, subpath: str)`
Инициализирует объект класса MainLogger и создает файл лога в указанной подпапке.

- `subpath`: Подпапка, в которой будет сохранен файл лога.
- `flog(self, message: str, is_print: bool = True, log_type: str = None)`: Записывает сообщение в лог.

  - `message`: Сообщение для записи в лог.
  - `is_print`: Флаг, указывающий, нужно ли выводить сообщение в консоль. По умолчанию True.
  - `log_type`: Тип логирования (например, 'info', 'warning', 'error'). По умолчанию None.


    class MainLogger:

        def __init__(self, subpath: str):
            """
            Инициализирует объект класса MainLogger.
            """
            agg_handler = logging.FileHandler(
                os.path.join(subpath, 'aggregation.log')
            )
            formatter = logging.Formatter(
                "%(asctime)s ----- %(levelname)s ---- %(threadName)s:    %(message)s"
            )
            agg_handler.setFormatter(formatter)
            self.logger = logging.getLogger('agg_logger')
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(agg_handler)
    
        def flog(self, message: str, is_print: bool = True, log_type: str = None):
            """
            Записывает сообщение в лог.
            """
            if is_print:
                print(f"{strftime('%Y-%m-%d %H:%M:%S', gmtime())}|   {message}")
            if log_type:
                log_function = getattr(self.logger, log_type)
                log_function(message)

Пример лога приведен ниже:

    2024-04-09 14:45:08,300 ----- INFO ---- MainThread: Агрегация запущена
    2024-04-09 14:46:28,402 ----- INFO ---- MainThread: Настройки источников обработаны
    2024-04-09 14:46:38,730 ----- INFO ---- MainThread: Обнаружено 1 файл(а) Metadata, в которых содержатся данные 8991 плавок
    2024-04-09 14:46:38,745 ----- INFO ---- MainThread: Метадата обработана
    2024-04-09 14:46:42,339 ----- INFO ---- MainThread: Обнаружено 38439 файлов источника LNK100
    ...
    2024-04-09 20:42:55,187 ----- INFO ---- MainThread: Создание маппинга завершено. Всего обнаружено 7823 уникальных заготовок
    2024-04-09 20:42:55,327 ----- INFO ---- MainThread: Мапинг создан. Всего 7823 уникальных заготовок
    2024-04-09 20:42:55,351 ----- INFO ---- MainThread: Заготовки распределены по 20 ядрам. Всего 392 итераций
    2024-04-09 20:42:55,376 ----- INFO ---- MainThread: Запуск аггрегации.

---

# 3. Чтение и парсинг настроек агрегации.

Из конфигурационного файла `settings.xlsx` считываются и парсятся настройки агрегации. Результат сохраняется в `sources_settings` структура которого 
указана в модуле `dto`.

    class MainPipeline:

      def run_pipeline(self):

      ...

      settings = pd.read_excel(
          "settings.xlsx", index_col=[0], header=None
      ).iloc[:, :-1]
      sources_settings = parse_settings(settings)

      ...

Пример настроек агрегации файла `settings.xlsx` приведены ниже:

| source             | LNK100                         | U0                                                                                                                                             | WBF_PIRO                                                                                | WBF_sgl                   | Имя фичи, будет использоваться как ключ колонки                                                                                                                                                                                                                                                                                 |
|--------------------|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| rolling_number     | 1                              | 1                                                                                                                                              | 1                                                                                       | 1                         | Номер проката заготовки                                                                                                                                                                                                                                                                                                         |
| handler            | BASE                           | BASE                                                                                                                                           | WBFPIRO                                                                                 | WBFSINGLE                 | Обработчик данных                                                                                                                                                                                                                                                                                                               |
| type               | target                         | feature                                                                                                                                        | feature                                                                                 | feature                   | Тип фичи                                                                                                                                                                                                                                                                                                                        |
| encoding           | UTF-8                          | UTF-8                                                                                                                                          | UTF-8                                                                                   | ANSI                      | Кодировка, в которой представлены данные                                                                                                                                                                                                                                                                                        |
| is_single          | FALSE                          | FALSE                                                                                                                                          | FALSE                                                                                   | TRUE                      | - Если атрибут в TRUE, это означает, что все данные по заготовкам собраны в одном CSV файле, и каждая строка в этом файле представляет отдельную заготовку. <br> - Если атрибут установлен в False, это означает, что данные по каждой заготовке хранятся в отдельных CSV файлах, а эти файлы собраны в папке `nested_folders`. |
| main_folder        | \\ZSMK-9684-001\Data\2023      | \\ZSMK-9684-001\Data\2023                                                                                                                      | \\ZSMK-9684-001\Data\2023                                                               | \\ZSMK-9684-001\Data\2023 | Корневая папка, откуда начинается рекурсивный поиск                                                                                                                                                                                                                                                                             |
| key_folder         | ipsrnk                         | U0                                                                                                                                             | wbf                                                                                     | wbf                       | Ключевая папка, в которой поиск ведется по пути nested_folders (если есть) и выбираются только файлы, содержащие filename_key (если есть)                                                                                                                                                                                       |
| nested_folders     | rail_points_files              | rollings_points_files                                                                                                                          | billet_pirometer_files                                                                  |                           | Путь внутри key_folder                                                                                                                                                                                                                                                                                                          |
| filename_key       |                                |                                                                                                                                                |                                                                                         |                           | Ключ файла                                                                                                                                                                                                                                                                                                                      |
| interpolation_type | by value                       | by value                                                                                                                                       |                                                                                         |                           | Тип интерполяции:<br>- by value - по указанному значению<br>- by source - по источнику                                                                                                                                                                                                                                          |
| interpolation      | 103                            | 103                                                                                                                                            |                                                                                         |                           | Интерполяция может быть:<br>1. Числовой - для интерполяции к определенному числу<br>2. По источнику - для интерполяции к длине указанного источника                                                                                                                                                                             |
| segments           | [1, 0, 3, []]                  | "[1, 0, 3, ['LNK100_1']]<br>[2, 3, 6, ['LNK100_1']]<br>[3, 30, 70, ['LNK100_1']]<br>[4, -3, "end", ['LNK100_1']]<br>[5, -6, -3, ['LNK100_1']]" | [1, 0, "end", ['LNK100_1']]                                                             | [1, 0, 0, ['LNK100_1']]   | Сегменты, на которые разбивается заготовка (важно! сегменты выделяются после интерполяции). Формат сегмента:<br>[идентификатор сегмента, начальная точка, конечная точка, [Имена целевых сегментов, к которым будет сопоставлен данный сегмент]]<br>Имя целевого сегмента формируется так: source_"идентификатор сегмента"      |
| billet_column      | BilletPoints                   | billet_points                                                                                                                                  | moment                                                                                  | billet_number             | Имя столбца с Billet points                                                                                                                                                                                                                                                                                                     |
| filtration_methods | ["std"]                        | ['std']                                                                                                                                        | []                                                                                      | []                        | Методы фильтрации столбцов                                                                                                                                                                                                                                                                                                      |
| secondary_features | ["abs"]                        | ['abs']                                                                                                                                        | []                                                                                      | []                        | Вспомогательные функции для обработки и анализа данных                                                                                                                                                                                                                                                                          |
| agg_methods        | ['tg', 'max', 'min', 'median'] | ['max', 'min','median','mean']                                                                                                                 | ['max', 'min','median','mean']                                                          | []                        | Методы агрегации данных внутри сегмента. К методу можно добавить настройку abs, используя нижнее подчеркивание. В этом случае данные сегмента будут предварительно взяты по модулю                                                                                                                                              |
| forbidden_columns  | []                             | ['moment']                                                                                                                                     | ["BEAM_VERT_ENC", "BEAM_HOR_ENC_POS", "DM_ENC_HOR_POS", "DML_VERT_POS", "DMR_VERT_POS"] | []                        | Имена столбцов, которые можно исключить из источника                                                                                                                                                                                                                                                                            |
| convert_columns    |                                |                                                                                                                                                |                                                                                         |                           |                                                                                                                                                                                                                                                                                                                                 |


#### Функция `parse_settings()`

Функция `parse_settings()` из модуля `source_parser` принимает на вход DataFrame типа `settings.xlsx`,
содержащий настройки агрегации, и парсит его в словарь объектов `SourceSettings`, где
ключами являются названия источников данных (агрегата).

Параметры

- settings (pd.DataFrame): DataFrame с настройками агрегации.

Возвращаемое значение

- Dict[str, SourceSettings]: Словарь объектов `SourceSettings`, где ключ - название источника данных (агрегата).


    def parse_settings(settings: pd.DataFrame) -> Dict[str, SourceSettings]:
        """
        Парсит настройки агрегации из DataFrame в словарь объектов SourceSettings.
        """
        sources = {}
        for _, source_settings in settings.iteritems():
            source_settings = source_settings.fillna("")
        ...
    
- Парсинг сегментов:

  - Для каждого источника данных из DataFrame происходит парсинг сегментов,
  содержащихся в строковом формате в столбце "segments".
  - Сегменты представлены в виде строки и разделены символом новой строки \n.
  Каждый сегмент имеет формат (id, start, end, target_ids) и парсится с помощью eval().
  - Созданные сегменты сохраняются в словарь segments, где ключи состоят из названия
  источника и идентификатора сегмента.

            ...
            # Парсим сегменты
            segments = {}
            for segment_data in str(source_settings["segments"]).split("\n"):
                segment_id, start, end, target_ids = eval(segment_data)
                segments[f"{source_settings['source']}_{segment_id}"] = Segment(
                    start_point=start, end_point=end, target_segments=target_ids
                )
            ...

- Парсинг специальных колонок:

  - Если в столбце "convert_columns" есть какие-то значения, то они также парсятся
  и добавляются в словарь convert_columns.

            ...
            # Парсим специальные колонки
            convert_columns = {}
            if source_settings["convert_columns"] != '':
                for convert_col in str(source_settings["convert_columns"]).split("\n"):
                    colname, convert_type = eval(convert_col)
                    convert_columns[colname] = convert_type
            ...    

- Парсинг остальных настроек источника:

  - Остальные настройки источника (тип, обработчик, кодировка и т. д.) парсятся и
  сохраняются в объект SourceSettings.

            ...
            # Парсим остальные настройки источника
            if source_settings['interpolation_type'] == 'by value':
                interp = float(source_settings['interpolation'])
            elif source_settings['interpolation_type'] == 'by source':
                interp = source_settings['interpolation']
            elif source_settings['interpolation_type'] == "":
                interp = None
            else:
                raise NameError(
                    f"Unknown interpolation type for source {source_settings['source']}"
                )
            ...

- Возврат результатов:

  - Все данные источника, включая сегменты и специальные колонки, сохраняются в объекте
  SourceSettings и добавляются в словарь sources, где ключ - название источника данных (агрегата).    

            ...
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
                forbidden_columns=ast.literal_eval(source_settings['forbidden_columns']),
                filtration_methods=ast.literal_eval(source_settings['filtration_methods']),
                secondary_features=ast.literal_eval(source_settings['secondary_features']),
                aggregation_methods=ast.literal_eval(source_settings['agg_methods']),
                convert_columns=convert_columns
            )
    
        return sources

  
Таким образом, после обработки настроек агрегации из файла `settings.xlsx` будет возвращен словарь следующего вида:

    {
        'LNK100': SourceSettings(
            source='LNK100',
            type='target',
            handler='BASE',
            rolling_number='1',
            encoding='UTF-8',
            is_single=False,
            main_folder='\\\\ZSMK-9684-001\\Data\\2023',
            key_folder='ipsrnk',
            nested_folders='rail_points_files',
            filename_key='',
            interpolation_type='by value',
            interpolation=103.0,
            segments={'LNK100_1': Segment(start_point=0, end_point=3, target_segments=[])},
            forbidden_columns=[],
            filtration_methods=['std'],
            secondary_features=['abs'],
            aggregation_methods=['tg', 'max', 'min', 'median'],
            billet_column='BilletPoints',
            convert_columns={}
        ),
        'U0': SourceSettings(
            source='U0',
            type='feature',
            handler='BASE',
            rolling_number='1',
            encoding='UTF-8',
            is_single=False,
            main_folder='\\\\ZSMK-9684-001\\Data\\2023',
            key_folder='U0',
            nested_folders='rollings_points_files',
            filename_key='',
            interpolation_type='by value',
            interpolation=103.0,
            segments={
                'U0_1': Segment(start_point=0, end_point=3, target_segments=['LNK100_1']),
                'U0_2': Segment(start_point=3, end_point=6, target_segments=['LNK100_1']),
                'U0_3': Segment(start_point=30, end_point=70, target_segments=['LNK100_1']),
                'U0_4': Segment(start_point=-3, end_point='end', target_segments=['LNK100_1']),
                'U0_5': Segment(start_point=-6, end_point=-3, target_segments=['LNK100_1'])
            },
            forbidden_columns=['moment'],
            filtration_methods=['std'],
            secondary_features=['abs'],
            aggregation_methods=['max', 'min', 'median', 'mean'],
            billet_column='billet_points',
            convert_columns={}
        ),
        'WBF_PIRO': SourceSettings(
            source='WBF_PIRO',
            type='feature',
            handler='WBFPIRO',
            rolling_number='1',
            encoding='UTF-8',
            is_single=False,
            main_folder='\\\\ZSMK-9684-001\\Data\\2023',
            key_folder='wbf',
            nested_folders='billet_pirometer_files',
            filename_key='',
            interpolation_type='',
            interpolation=None,
            segments={'WBF_PIRO_1': Segment(start_point=0, end_point='end', target_segments=['LNK100_1'])},
            forbidden_columns=['BEAM_VERT_ENC', 'BEAM_HOR_ENC_POS', 'DM_ENC_HOR_POS', 'DML_VERT_POS', 'DMR_VERT_POS'],
            filtration_methods=[],
            secondary_features=[],
            aggregation_methods=['max', 'min', 'median', 'mean'],
            billet_column='moment',
            convert_columns={}
        ),
        'WBF_sgl': SourceSettings(
            source='WBF_sgl',
            type='feature',
            handler='WBFSINGLE',
            rolling_number='1',
            encoding='ANSI',
            is_single=True,
            main_folder='\\\\ZSMK-9684-001\\Data\\2023',
            key_folder='wbf',
            nested_folders='',
            filename_key='',
            interpolation_type='',
            interpolation=None,
            segments={'WBF_sgl_1': Segment(start_point=0, end_point=0, target_segments=['LNK100_1'])},
            forbidden_columns=[],
            filtration_methods=[],
            secondary_features=[],
            aggregation_methods=[],
            billet_column='billet_number',
            convert_columns={}
        )
    }


### Метод `_add_target_paths` класса `MainPipeline`

    ...
    self._add_target_paths(sources_settings)
    ...

Метод `_add_target_paths` класса `MainPipeline` используется для добавления путей к файлам и папкам, где будут сохранены результаты агрегации.
Дополнительные пути указаны в модуле `constants` переменных `LOGS_FOLDER`, `INITIAL_DATA_FOLDER`, `PREPARED_DATA_FOLDER`.


    class MainPipeline
    
    ...

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

---

# 4. Парсинг метаинформации.

Создается объект `creator` класса `MappingCreator`, который используется для парсинга метаданных.
Метаданные считываются и обрабатываются, результат парсинга, содержащий информацию о заготовках,
сохраняется в переменной `metadata` в виде DataFrame.

    class MainPipeline:

      def run_pipeline(self):

      ...

      creator = data_mapping.MappingCreator(setup=self.setup, logger=logger)
      metadata = creator.get_metadata()

      ...

### Class `MappingCreator`

Класс `MappingCreator` расположен в модуле `data_mapping`. При инициализации объект типа `MappingCreator` в качестве
атрибутов принимает настройки пайплайна и объект логгера.

    class MappingCreator:
    
        def __init__(self, setup: dto.PipelineSetup, logger: MainLogger):
            self.setup = setup
            self.logger = logger
            ...

#### Метод `get_metadata()`

Функция `get_metadata` предназначена для чтения и парсинга метаданных, содержащих информацию о плавках.


Пример файла метаданных о плавках приведен ниже:

| Серия     | №        | Плавка      | Марка  | Вр.откр.СК       | Руч     | Заг    | Заказ, м | Вр.реза          | Задано, мм | Факт, мм | По обж., мм | Рег-р, мм | Вр.взвеш-я       | Длина, мм  | Цель, кг  | Факт, кг   | Рег-р, мм | Вр.посадки       | Длина, мм  | Вес, кг    | Вр.проката       | Раскат, мм | Дл. МУЗК, мм | Дл. ЛНК, мм | Дл. СОС, мм |
|-----------|----------|-------------|--------|------------------|---------|--------|----------|------------------|------------|----------|-------------|-----------|------------------|------------|-----------|------------|-----------|------------------|------------|------------|------------------|------------|--------------|-------------|-------------|
| 1-23-0424 | 22       | Э02-23-9403 | 76ХФ   | 30.12.2023 13:59 | 1       | 4      | 7,93     | 30.12.2023 15:36 | 7952       | 7953     | 11          | -88       | 30.12.2023 15:42 |            | 6730      | 6768       | -0,5      | 01.01.2024 16:19 | 7846       | 6755       | 01.01.2024 20:50 | 105580     |              | 103228      | 101508      |
| 1-23-0424 | 22       | Э02-23-9403 | 76ХФ   | 30.12.2023 13:59 | 2       | 1      | 7,93     | 30.12.2023 15:01 | 8056       | 8057     | -20         | 36        | 30.12.2023 15:08 |            | 6730      | 6742       | 0         | 01.01.2024 16:20 | 7950       | 6725       | 01.01.2024 20:52 | 105050     |              | 103081      | 101483      |
| 1-23-0424 | 22       | Э02-23-9403 | 76ХФ   | 30.12.2023 13:59 | 2       | 4      | 7,93     | 30.12.2023 15:37 | 8035       | 8036     | -37         | 32        | 30.12.2023 15:44 |            | 6730      | 6730       | 0         | 01.01.2024 16:22 | 7913       | 6720       | 01.01.2024 20:54 | 104990     |              | 103061      | 103108      |
| 1-23-0424 | 22       | Э02-23-9403 | 76ХФ   | 30.12.2023 13:59 | 3       | 1      | 7,93     | 30.12.2023 15:01 | 8043       | 8044     | -41         | 44        | 30.12.2023 15:10 |            | 6730      | 6734       | 0         | 01.01.2024 16:24 | 7912       | 6720       | 01.01.2024 20:56 | 104900     |              | 103034      | 101432      |
| 1-23-0424 | 22       | Э02-23-9403 | 76ХФ   | 30.12.2023 13:59 | 4       | 1      | 7,93     | 30.12.2023 15:01 | 8057       | 8058     | 1           | 16        | 30.12.2023 15:12 |            | 6730      | 6726       | 0         | 01.01.2024 16:25 | 7954       | 6715       | 01.01.2024 20:58 | 104920     |              | 103047      | 101438      |
| 1-23-0424 | 22       | Э02-23-9403 | 76ХФ   | 30.12.2023 13:59 | 3       | 4      | 7,93     | 30.12.2023 15:37 | 8032       | 8033     | -52         | 44        | 30.12.2023 15:46 |            | 6730      | 6726       | 0         | 01.01.2024 16:27 | 7923       | 6715       | 01.01.2024 21:00 | 104830     |              | 102966      | 103010      |
| 1-23-0424 | 22       | Э02-23-9403 | 76ХФ   | 30.12.2023 13:59 | 4       | 4      | 7,93     | 30.12.2023 15:37 | 8043       | 8044     | -10         | 12        | 30.12.2023 15:48 |            | 6730      | 6730       | 0         | 01.01.2024 16:28 | 7944       | 6720       | 01.01.2024 21:02 | 104960     |              | 103066      | 101454      |
| ...       | ...      | ...         | ...    | ...              | ...     | ...    | ...      | ...              | ...        | ...      | ...         | ...       | ...              | ...        | ...       | ...        | ...       | ...              | ...        | ...        | ...              | ...        | ...          | ...         | ...         |


    class MappingCreator:

      def get_metadata(self):
          billets_info = list()

- Извлечение списка файлов метаданных из указанного пути `self.setup.PATH_TO_METADATA`.

          ...

          metadata_files = glob(self.setup.PATH_TO_METADATA)

          ...  

- Проверяет, есть ли файлы метаданных. Если их нет, генерирует ошибку.

          ...  

          if not metadata_files:
              error_msg = "Не найдено файлов Metadata. Процесс остановлен"
              self.logger.flog(error_msg, is_print=False, log_type="error")
              raise FileNotFoundError(error_msg)
  
          ...

- Для каждого файла метаданных читает его содержимое в объект DataFrame с помощью `pd.read_excel()`.

          ...

          for metadata_path in glob(self.setup.PATH_TO_METADATA):
              metadata = pd.read_excel(metadata_path, engine='openpyxl')

          ...


- Добавляет новый столбец BilletId, содержащий уникальный идентификатор заготовки, сгенерированный
на основе данных из других столбцов.

              ...

              metadata[self.setup.METADATA_BILLET_ID] = \
                  "Л2" + \
                  metadata["Плавка"].str.split("-", expand=True).iloc[:, -1] + \
                  metadata["Руч"].astype(str) + "0" + \
                  metadata["Заг"].astype(str) + "_" + \
                  metadata["Вр.реза"].dt.year.astype(str)

              ...

- Если включен фильтр по маркам `MARK_FILTER`, фильтрует метаданные по указанным маркам.

              ...

              if self.setup.MARK_FILTER is True:
                  metadata = metadata[metadata["Марка"].isin(self.setup.MARK)]

              ...

- Извлекает необходимую информацию о плавках (дата проката, идентификатор заготовки, марка) и
сохраняет ее в отдельном DataFrame.

            ...

            billets_info.append(
                metadata[["Вр.проката", "BilletId", "Марка"]].dropna()
            )

            ...

- Объединяет информацию о плавках из всех файлов метаданных в один DataFrame.

            ...

            billets_metadata = pd.concat(billets_info)
            self.logger.flog(
                f"Обнаружено {len(metadata_files)} файл(а) Metadata, "
                f"в которых содержатся данные {billets_metadata.shape[0]} "
                f"плавок",
                is_print=False,
                log_type="info"
            )

            ...

- Удаляет дубликаты строк на основе столбца BilletId. Возвращает объединенный DataFrame с уникальными плавками.

            ...

            return pd.concat(billets_info).drop_duplicates(subset=["BilletId"])


В результате выполнения данной функции будет возвращен DataFrame по уникальным плавкам (дата проката, идентификатор заготовки, марка), 
которые затем используются для соотнесения данных из различных источников.

Пример полученного DataFrame по уникальным заготовкам приведен ниже:

| Вр.проката       | BilletId         | Марка  |
|------------------|------------------|--------|
| 2023-04-01 04:37 | Л22952101_2023   | Э76ХФ  |
| 2023-04-01 04:39 | Л22952105_2023   | Э76ХФ  |
| 2023-04-01 04:41 | Л22952201_2023   | Э76ХФ  |
| 2023-04-01 04:43 | Л22952205_2023   | Э76ХФ  |
| 2023-04-01 04:45 | Л22952301_2023   | Э76ХФ  |
| ...              | ...              | ...    |
| 2023-10-31 07:19 | Л27948204_2023   | 76ХФ   |
| 2023-10-31 07:22 | Л27948301_2023   | 76ХФ   |
| 2023-10-31 07:24 | Л27948401_2023   | 76ХФ   |
| 2023-10-31 07:27 | Л27948304_2023   | 76ХФ   |
| 2023-10-31 07:29 | Л27948404_2023   | 76ХФ   |

---

# 5. Создание мапинга файлов.

Вызов метода `create_mapping()` для создания маппинга файлов на основе полученной
информации о заготовках и настроек агрегации. В результате создания маппинга формируются две структуры
данных: словарь `by_billets`, где каждая заготовка ассоциируется с соответствующим
набором файлов данных, и датафрейм `not_use_billets`, содержащий информацию о
заготовках, для которых не удалось найти соответствующие файлы данных.

### Типы файлов выгрузок

Ниже приведена общая структура корневой папки, с которой стартует рекурсивный поиск:

    └── корневая папка (source.main_folder)
        ├── 01 (месяц)
        |    ├── 01 (день)
        |    |    └── ключевая папка (source.key_folder)
        |    |        ├── сводный файл прокатов 
        |    |        ├── сводный файл одиночных сигналов 
        |    |        └── папка с файлами, значения которых привязаны к длине заготовки (source.nested_folders)
        |    |            ├── файл с данными по первому прокату 
        |    |            ├── файл с данными по второму прокату
        |    |            └── ...
        |    |
        |    ├── 02 (день)
        |    |    └── ...
        |    └── ...
        |
        ├── 02 (месяц)
        |   ├── 01 (день)
        |   |    └── ...
        |   └── ...
        └── ...


    └── \ZSMK-9684-001\Data\2023 (source.main_folder)
        ├── 01 (месяц)
        |    ├── 01 (день)
        |    |    ├── BD1
        |    |    |    └── ...
        |    |    ├── BD2
        |    |    |    └── ...
        |    |    ├── U0 (source.key_folder)
        |    |    |    ├── 20230317140000_20230317150000_U0.csv
        |    |    |    ├── 20230317140000_20230317150000_U0_S.csv
        |    |    |    └── rollings_points_files (source.nested_folders)
        |    |    |        ├── 20230317140142_Л227544011_U0_1_L.csv
        |    |    |        ├── 20230317140346_Л227543041_U0_1_L.csv
        |    |    |        └── ...
        |    |    ├── wbf
        |    |    |    └── ...
        |    |    └── ...
        |    |
        |    ├── 02 (день)
        |    |    └── ...
        |    └── ...
        |
        ├── 02 (месяц)
        |   ├── 01 (день)
        |   |    └── ...
        |   └── ...
        └── ...


### Метод `create_mapping()`

#### 1) Инициализация переменных: Создаются переменные для хранения информации о файлах и заготовках.


    sources = dict() # словарь, где каждому источнику данных соответствует словарь с идентификаторами заготовок и соответствующими им путями к файлам

    by_billets = dict() # словарь, где ключ - идентификатор заготовки, а значение - словарь, содержащий пути к файлам для каждого источника данных.

    not_use_billets = DataFrame() # информация о заготовках, которые не были использована в маппинге.


#### 2) Инициализация поиска директорий по источникам данных (агрегата): Для каждого источника данных (агрегата), указанного в настройках, происходит поиск соответствующих ключевых папок


    class MappingCreator:

          def create_mapping(self, metadata: pd.DataFrame, settings: Dict[str, SourceSettings]) -> Tuple[dict, pd.DataFrame]:
            ...
        
            for source_name, source in settings.items():
        
            ...

Пример `source_name` приведен ниже:

-    "WBF_PIRO"


Пример `source` приведен ниже:


    SourceSettings(
                source="WBF_PIRO",
                type="feature",
                handler="WBFPIRO",
                rolling_number="1",
                encoding="UTF-8",
                is_single=False,
                main_folder=r"\\ZSMK-9684-001\Data\2023",
                key_folder="wbf",
                nested_folders="billet_pirometer_files",
                filename_key="",
                interpolation_type="",
                billet_column="moment",
                interpolation=None,
                segments={
                    "WBF_PIRO_1": Segment(start_point=0, end_point="end", target_segments=["LNK100_1"])
                },
                forbidden_columns=["BEAM_VERT_ENC", "BEAM_HOR_ENC_POS", "DM_ENC_HOR_POS", "DML_VERT_POS", "DMR_VERT_POS"],
                filtration_methods=[],
                secondary_features=[],
                agg_methods=["max", "min", "median", "mean"],
                convert_columns={}
            )

#### 3) Поиск ключевых папок по агрегату: Для каждого источника данных (агрегата) вызывается метод `_get_key_folders` для получения путей до ключевых папок.


    class MappingCreator:

          def create_mapping(self, metadata: pd.DataFrame, settings: Dict[str, SourceSettings]) -> Tuple[dict, pd.DataFrame]:
            ...
        
            files_folders = self._get_key_folders(
                main_folder=source.main_folder, key=source.key_folder
            )
        
            ...


#### Метод `_get_key_folders()`

Метод `_get_key_folders` проходит через иерархию каталогов, начиная с `source.main_folder`, и ищет папки с именем `source.key_folder`.
Как только такая папка найдена, функция возвращает путь к ней.


    class MappingCreator:

      ...

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

После того, как найдены все пути до ключевых папок по источнику данных (агрегату),
к каждому из этих путей присоединяется `source.nested_folders` (таким образом формируется путь до папки с файлами по каждой заготовке)
и затем к этому пути добавляется маска для поиска файлов (*csv).
Таким образом, происходит поиск всех CSV-файлов внутри каждой папки, связанной с `source.key_folder` и `source.nested_folders`.

    class MappingCreator:

          def create_mapping(self, metadata: pd.DataFrame, settings: Dict[str, SourceSettings]) -> Tuple[dict, pd.DataFrame]:
            ...
        
            for folder in files_folders:
                filepaths.extend(
                    glob(
                        os.path.join(
                            folder, f"{source.nested_folders}", "*csv"
                        )
                    )
                )
        
            ...

#### 4) Фильтрация полученных путей до файлов по `source.filename_key`

Проверяем каждый путь к файлу и добавляем его в список `filtered_filepaths` только в том случае,
если ключевая строка `source.filename_key` содержится в имени файла. Если же такой строки нет в имени файла,
то данный путь не добавляется в список `filtered_filepaths` и, следовательно, не участвует в дальнейшей обработке.

    class MappingCreator:

          def create_mapping(self, metadata: pd.DataFrame, settings: Dict[str, SourceSettings]) -> Tuple[dict, pd.DataFrame]:
            ...
        
            filtered_filepaths = []
            for filepath in filepaths:
                if source.filename_key in filepath:
                    filtered_filepaths.append(filepath)
        
            ...

Если файлы найдены, то производится запись об этом в лог.

#### 5) Обработка полученных путей к файлам. Создание словаря `sources`

    class MappingCreator:

          def create_mapping(self, metadata: pd.DataFrame, settings: Dict[str, SourceSettings]) -> Tuple[dict, pd.DataFrame]:
            ...
        
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
        
            ...

Если атрибут `source.is_single` установлен в True, это означает, что все данные по заготовкам собраны в одном CSV файле,
и каждая строка в этом файле представляет отдельную заготовку. В этом случае функция `_get_single_billet_id` читает содержимое
CSV файла и для каждой строки проверяет, существует ли соответствие идентификатора заготовки в метаданных.

    class MappingCreator:
            ...
        
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

Если идентификатор заготовки найден и время проката находится в заданном диапазоне, что проверяется функцией
`_is_billet_in_period`, тогда формируется кортеж `(billet_data["BilletId"].iloc[0], filepath)` (идентификатор заготовки, путь к файлу)
и добавляется в список `filepath_mapping`.

    class MappingCreator:
            ...

            def _is_billet_in_period(self, billet_time) -> bool:
                return self.setup.MIN_TIME <= billet_time <= self.setup.MAX_TIME

Если атрибут `source.is_single` установлен в False, это означает, что данные по каждой заготовке хранятся в отдельных CSV файлах,
а эти файлы собраны в папке `source.nested_folders`. В этом случае вызывается функция `_get_billet_id`, которая проверяет базовое имя файла
на наличие в нем `_L` или `_T`, что указывает на то что это файл с данными сигналов, обработанный по модели нарезки значений сигнала по длине 
или по времени. Затем по имени файла формируется ID заготовки `billet_id` с помощью регулярных выражений, основанных на шаблонах, таких как `Л\d{8}`,
где `\d{8}` - восьмизначное число, и символ `Л`.

    class MappingCreator:
            ...

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

Затем вызывается функция `_get_billet_from_period` для того чтобы перевести первые 14 символов имени файла `file_name[:14]` в datetime формат, 
так как первые 14 символов имени файла обозначают время начала проката: `{время начала проката}_{номер заготовки}_{агрегат}_{номер проката}_L.csv`.

    class MappingCreator:
            ...

            def _get_billet_from_period(self, date: str, billet_id: str, filepath):
                billet_time = datetime.strptime(date, "%Y%m%d%H%M%S")
                if self._is_billet_in_period(billet_time):
                    return f"{billet_id}_{billet_time.year}", filepath

После этого функцией `_is_billet_in_period` проверяется соответствие времени начала проката заготовки заданному диапазону времени.
Если время начала проката удовлетворяет условиям диапазона и идентификатор заготовки соответствует регулярному выражению, то заготовке присваивается
уникальный идентификатор в формате `f"{billet_id}_{billet_time.year}"`, который затем добавляется в словарь `sources` вместе с путем к файлу.
Если идентификатор заготовки не удовлетворяет условиям регулярных выражений или временного диапазона, функция возвращает `None`.

Результаты этих функций затем собираются в словарь `sources`, где каждому источнику данных соответствует словарь с идентификаторами
заготовок и соответствующими им путями к файлам. Пример приведен ниже:


    sources = {
        'LNK100': {
            'Л12345678_2023': '\\ZSMK-9684-001\Data\2023\01\01\LNK100\rollings_points_files\20230317140346_Л227543041_U0_1_L.csv',
            'Л87654321_2023': '\\ZSMK-9684-001\Data\2023\01\01\LNK100\rollings_points_files\20230317140346_Л227543041_U0_1_L.csv',
            ...
        },
        'U0': {
            'Л23456789_2023': '\\ZSMK-9684-001\Data\2023\08\09\U0\rollings_points_files\20230809113255_Л258973020_U0_1_L.csv',
            'Л98765432_2023': '\\ZSMK-9684-001\Data\2023\08\09\U0\rollings_points_files\20230809113255_Л258973020_U0_1_L.csv',
            ...
        },
        'WBF_PIRO': {
            'Л34567890_2023': '\\ZSMK-9684-001\Data\2023\12\14\wbf\billet_pirometer_files\20231214172621_23-Л211344030_WBF_1_T2.csv',
            'Л54321098_2023': '\\ZSMK-9684-001\Data\2023\12\14\wbf\billet_pirometer_files\20231214172621_23-Л211344030_WBF_1_T2.csv',
            ...
        },
        ...
    }


#### 6) Обработка полученного словаря `sources`

    class MappingCreator:
            ...

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


Создается пустой словарь `by_billets`, который будет содержать информацию о каждой уникальной заготовке и ее соответствующих файлов.

    class MappingCreator:
            ...

            by_billets = dict()

Создается словарь `not_use_billets`, который будет содержать информацию о заготовках, для которых не удалось найти
соответствующие файлы в каждом из источников данных.

    class MappingCreator:
            ...

            not_use_billets = {key: [] for key in sources.keys()}

В цикле перебираются идентификаторы заготовок из метаданных.

    class MappingCreator:
            ...

            for billet_id in metadata[self.setup.METADATA_BILLET_ID].to_list():

Для каждого идентификатора заготовки проверяется, есть ли эта заготовка в каждом из источников данных `sources`.
Если да, то создается запись в словаре `by_billets`, где ключ - идентификатор заготовки, а значение - словарь, содержащий пути к
файлам для каждого источника данных.

    class MappingCreator:
            ...

            if all([billet_id in billet_ids for _, billet_ids in sources.items()]):
                by_billets[billet_id] = dict(
                    [
                        (source, sources[source][billet_id])
                        for source in sources
                    ]
                )

Если для какой-либо заготовки не удалось найти соответствующие файлы во всех источниках данных,
информация о ней добавляется в словарь `not_use_billets`.

    class MappingCreator:
            ...

            else:
                not_use_billets["BilletId"].append(billet_id)
                not_use_billets["Time"].append(
                    metadata[metadata[self.setup.METADATA_BILLET_ID] ==
                             billet_id]["Вр.проката"].iloc[0]
                )
                for source, billet_ids in sources.items():
                    is_in_billet_ids = int(billet_id in billet_ids)
                    not_use_billets[source].append(is_in_billet_ids)

В конце функция форматирует данные о найденных заготовках и выводит их в виде словаря `by_billets`,
где ключ - идентификатор заготовки, а значение - словарь, содержащий пути к файлам для каждого источника данных:

    {
        "Л54321098_2023": {
            "LNK100": "\\ZSMK-9684-001\Data\2023\07\07\ipsrnk\rail_points_files\20230707001909_25178X302_IPSRNK_1_L.csv",
            "U0": "\\ZSMK-9684-001\Data\2023\08\09\U0\rollings_points_files\20230809113255_Л258973020_U0_1_L.csv",
            ...
        },
        "Л98765432_2023": {
            "WBF_PIRO": "\\ZSMK-9684-001\Data\2023\12\14\wbf\billet_pirometer_files\20231214172621_23-Л211344030_WBF_1_T2.csv",
            "U0": "\\ZSMK-9684-001\Data\2023\08\09\U0\rollings_points_files\20230809113255_Л258973020_U0_1_L.csv",
            ...
        },
        ...
    }

А также выводит информацию о не найденных заготовках во всех источниках в виде таблицы `not_use_billets`:


    {
        'LNK100': [0, 0, 1, ...],  # Пример заготовки не использованной в источнике 'LNK100'
        'U0': [1, 0, 1, ...],       # Пример заготовки не использованной в источнике 'U0'
        'WBF_PIRO': [0, 1, 1, ...], # Пример заготовки не использованной в источнике 'WBF_PIRO'
        'BilletId': ['Л12345678_2023', 'Л23456789_2023', ...],  # Идентификаторы неиспользованных заготовок
        'Time': ['2024-05-01 12:00:00', '2024-05-02 13:00:00', ...],  # Время проката неиспользованных заготовок
        'fullness': ['1/3', '2/3', ...]  # Степень заполненности заготовок в различных источниках данных
    }

Здесь каждая строка представляет собой информацию о заготовке, которая не была использована в маппинге.
Колонки `LNK100`, `U0`, `WBF_PIRO` указывают, присутствует ли соответствующая заготовка в соответствующем источнике данных.
Столбец `fullness` показывает степень заполненности заготовок в различных источниках данных.

| Time                | BilletId         | LNK100 | U0  | WBF_PIRO | fullness |
|---------------------|------------------|--------|-----|----------|----------|
| 2024-05-01 12:00:00 | Л12345678_2023   | 0      | 1   | 0        | 1/3      |
| 2024-05-02 13:00:00 | Л23456789_2023   | 1      | 0   | 1        | 2/3      |
| ...                 | ...              | ...    | ... | ...      | ...      |

---

# 6. Определение компонент запуска процесса обработки файлов

### Определение вспомогательных компонент `Materials`

Если при настройке пайплайна указан путь к материалам `PATH_TO_MATERIALS`, то создается словарь `PATHS`,
где ключами являются имена файлов без расширения, а значениями - пути к файлам.

    materials = Materials(
        {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in glob(self.setup.PATH_TO_MATERIALS)
        } if self.setup.PATH_TO_MATERIALS else None
    )

Тогда словарь `materials` будет иметь следующий вид:

    materials = {
        'max_data': 'agregator/run/materials/max_data.json',
        'min_data': 'agregator/run/materials/min_data.json',
        'signal_lists': 'agregator/run/materials/signal_lists.csv',
        'wbf_piro_cutter_settings': 'agregator/run/materials/wbf_piro_cutter_settings.json',
        'workcenters': 'agregator/run/materials/workcenters.json'
    }

В папке `materials` указываются такие файлы как:

- `max_data.json`: Файл JSON с максимальными значениями параметров:


    {
        "LNK100_Vert1000": 3519.0,
        "LNK100_Vert1500": 4409.0,
        "LNK100_Vert2000": 4438.0,
        "LNK100_Vert3000": 5102.0,
        "LNK100_Hor1000": 1928.0,
        "LNK100_Hor1500": 3197.0,
        "LNK100_Hor2000": 4720.0,
        "LNK100_Torsion": 1835.0,
        "U0_u0.descaling.pressure.extrapolated": 155.85667419433594,
        "U0_u0.gr214_bt01_temp.extrapolated": 941.8836669921876,
        "U0_u0.ds.pos_low_hor_roll": 131.69558715820312,
        "U0_u0.os.pos_low_hor_roll": 131.6527862548828,
    ...
    }


- `min_data.json`: Файл JSON с минимальными значениями параметров:


    {
      "LNK100_Torsion": -4998.0,
      "LNK100_classification_Torsion": 0.0,
      "LNK100_abs_Torsion": 0.0,
      "TBK_h": 21.82,
      "TBK_tM": -9024.85,
      "TBK_tF": -7777.48,
      "TBK_fM": -8670.08,
      "TBK_fF": -9150.54,
      "TBK_wF": 1.94,
    ...
    }


- `signal_lists.csv`: Файл CSV с перечнем сигналов:


    whitelist;blacklist
    e.gh31.pos.set;BEAM_HOR_ENC
    e.reference_inclination_horizontal_top_roll;speed
    temp.average;
    temp.max;
    u0.gr212_1.ds.table_pos_before;
    u0.gr232_2.ds.table_pos_after;
    u0.os.reference_thickness_vertical;
    u0.reference_inclination_horizontal_top_roll;
    u0.reference_offset_axial_shifting;
    uf.ds.reference_thickness_vertical;
    uf.gr32-1.ds.table_pos_after.act;
    ...


- `wbf_piro_cutter_settings.json`: Файл JSON с настройками пирометров ПШБ WBF PIRO.


    {
        "PIROMETERS": ["TEMP_PIR1", "TEMP_PIR2", "TEMP_PIR3"],
        "SETTINGS": 
        {
                "BEAM_VERT_ENC": 
                [
                    [-0.1, 0.1, 20, 0.5, 10, 5],
                    [0.5, 10, 5, -0.1, 0.1, 20],
                    [-0.1, 0.1, 20, -10, -1, 5],
                    [-10, -1, 5, -0.1, 0.1, 20],
                    [-0.1, 0.1, 20, 0.5, 10, 5]
                ],
                "BEAM_HOR_ENC_POS": 
                [
                    [-0.1, 0.1, 20, 0.5, 10, 5],
                    [5, 50, 100, -0.1, 0.1, 20],
                    [-1, 1, 20, -10, -1, 5],
                    [-10, -1, 20, -0.1, 0.1, 20]
                ],
                ...
    }

- `workcenters.json`: Файл JSON с информацией по каждому источнику данных (агрегату):


    {
      "LNK100": "mill",
      "U0": "mill",
      "WBF_PIRO": "mill",
      ...
    }


### Компоненты запуска процесса обработки файлов

Компоненты запуска процесса обработки файлов делятся на следующие основные классы:

- `features_generator` class
- `secondary_functions` class
- `filters` class
- `segments_aggregator` class

Далее рассмотрен разбор классов, приведенных выше.

Рассмотрим классы `Transcription` и `Array`, так как класс `FeaturesGenerator` является зависимым от них.


### Transcription Class

`Transcription` - представляет собой объект, который необходим для того чтобы формировать идентификатор преобразованного сигнала.
Он хранит информацию о различных атрибутах данных, таких как: `workcenter`, `workunit`, `rolling_number`,
`name`, `model`, `interpolation`, `sector_range`, `preprocessing`, `aggregation` и `secondary_functions`.


#### Принцип формирования идентификатора преобразованного сигнала

#### Определения 

`workcenter` - рабочий блок (`mill` (стан), `ccm` (МНЛЗ)); 
 
`workunit` - рабочий юнит (агрегат (`bd1`, `e`, `uf`); 
 
`rolling_number` - номер прохода; 
 
`name` - название исходного сигнала, которое формируется по следующим правилам:
 
- Приведение всех символов к нижнему регистру; 
- При разделении слов регистрами ставить точку.


    Пример: 
    Было: GE11-1.DS.LineBefore.Act 
    Стало: ge11-1.ds.line.before.act


`model` - модель нарезки значений сигнала: 
 
- По длине: l1, l2 ... ln 
- По времени: t1, t2 ... tn 

`interpolation` - значение интерполяции. Если необходима интерполяция - `i{value}`, где `value` - число,
к которому интерполируются длина заготовки. Если интерполяция не нужна - `ni`; 
 
`sector_range` - область определения точек на заготовке: 
 
- `all` - берутся все точки на заготовке; 
- `h` + интервал - берутся точки на заготовке за указанный интервал, начиная с головы заготовки `(h5.0-6.5)`; 
- `t` + интервал - берутся точки на заготовке за указанный интервал, начиная с хвоста заготовки `(t5.0-6.5)`. 

`preprocessing` - метод предобработки каждого значения сигнала: 
 
- `abs` - приведение каждого значения к абсолютному. 

`aggregation` - метод расчёта значения сигнала (методы агрегации).

`secondary_functions` - вспомогательные функции агрегации.


Примеры:

Шаблон формирования идентификатора преобразованного сигнала:

    {workcenter}_{workunit}_{rolling_number}_{name}_{model}_{interpolation}_{sector_range}_{preprocessing}.{aggregation}

Стан, клеть bd1, сигнал temp, нарезанный по модели l1 - интерполируем длину заготовки после первого прохода клети к 103 метрам,
и находим минимальное значение сигнала на первых 3 метрах заготовки, начиная с головы:

    mill_bd1_1_temp_l1_i103.0_h0.0-3.0_min

Стан, клеть bd1, сигнал temp, нарезанный по модели l1 - не интерполируем длину заготовки после первого прохода клети, и находим максимальное значение сигнала:

    mill_bd1_1_temp_l1_ni_all_max

Ниже приведен разбор класса `Transcription`:

    class Transcription:
    
        def __init__(
            self,
            workcenter: str = "",
            workunit: str = "",
            rolling_number: str = "",
            name: str = "",
            model: str = "",
            interpolation: str = "",
            sector_range: str = "",
            preprocessing: str = "",
            aggregation: str = "",
            secondary_functions: str = ""
        ):
            self.workcenter = workcenter
            self.workunit = workunit
            self.rolling_number = rolling_number
            self.name = name
            self.model = model
            self.interpolation = interpolation
            self.sector_range = sector_range
            self.preprocessing = preprocessing
            self.aggregation = aggregation
            self.secondary_functions = secondary_functions

- Пример инициализации объекта класса `Transcription`:


    transcription = Transcription(
        workcenter="mill",
        workunit="bd1",
        rolling_number="1",
        name="temp",
        model="l1",
        interpolation="i103.0",
        sector_range="h0.0-3.0",
        preprocessing="ni",
        aggregation="min",
        secondary_functions=""
    )
    
    print(transcription)

- Вывод:


    mill_bd1_1_temp_l1_i103.0_h0.0-3.0_ni.min

#### Метод `__repr__`
Метод `__repr__` возвращает строковое представление объекта, включая значения всех его атрибутов.

        def __repr__(self):
            return str(self)

- Пример:


    transcription.__repr__()

- Вывод:


    'mill_bd1_1_temp_l1_i103.0_h0.0-3.0_ni.min'

#### Метод `__hash__`
Метод `__hash__` возвращает хеш-значение объекта, используемое для сравнения объектов в структурах данных,
таких как словари и множества.


        def __hash__(self):
            return hash(str(self))
    
- Пример:


    transcription.__hash__()

- Вывод:


    7894055372663120742

#### Метод `__eq__`
Метод `__eq__` проверяет, равен ли текущий объект другому объекту класса `Transcription`.


        def __eq__(self, other):
            return str(self) == str(other)
    
#### Метод `__str__`
Метод `__str__` возвращает строковое представление объекта, исключая пустые атрибуты и используя символы
подчеркивания и точек для разделения значений атрибутов.


        def __str__(self):
            return (
                f"{self._add_under(self.workcenter)}"
                f"{self._add_under(self.workunit)}"
                f"{self._add_under(self.rolling_number)}"
                f"{self._add_under(self.name)}"
                f"{self._add_under(self.model)}"
                f"{self._add_under(self.interpolation)}"
                f"{self._add_under(self.sector_range)}"
                f"{self._add_dot(self.preprocessing)}"
                f"{self._add_under(self.aggregation)}"
            )[:-1]


- Пример:


    transcription.__str__()

- Вывод:


    'mill_bd1_1_temp_l1_i103.0_h0.0-3.0_ni.min'

#### Метод `_add_under`
Метод `_add_under` добавляет символ подчеркивания к строке, если строка не пустая.


        @staticmethod
        def _add_under(string: str):
            return f"{string}_" if string != "" else ""

#### Метод `_add_dot`
Метод `_add_dot` добавляет точку к строке, если строка не пустая.


        @staticmethod
        def _add_dot(string: str):
            return f"{string}." if string != "" else ""

#### Метод `add_tags`
Метод `add_tags` используется для добавления тегов к идентификатору сигнала.


        def add_tags(self, tags: Dict[str, List[str]], replace: bool = False):
            new_transcription = copy(self)
            for tag_type, tags_values in tags.items():
                if replace:
                    full_tag = tags_values[0]
                    for tag in tags_values[1:]:
                        full_tag += new_transcription._add_under(tag)
                    setattr(new_transcription, tag_type, full_tag)
                else:
    
                    for tag in tags_values:
                        transcription_attr = getattr(new_transcription, tag_type)
                        setattr(
                            new_transcription,
                            tag_type,
                            f"{new_transcription._add_under(transcription_attr)}"
                            f"{tag}",
                        )
            return new_transcription

- Пример без замены (`replace=False`):


    new_transcription = transcription.add_tags(
        {
        "sector_range": ["t0.0-3.0"],
        "aggregation": ["max"]
    }, replace=False)
    print(new_transcription)

- Вывод:


    Было:
    mill_bd1_1_temp_l1_i103.0_h0.0-3.0_ni.min
    Стало:
    mill_bd1_1_temp_l1_i103.0_h0.0-3.0_t0.0-3.0_ni.min_max

- Пример с заменой (`replace=True`):


    new_transcription = transcription.add_tags(
        {
        "sector_range": ["t0.0-3.0"],
        "aggregation": ["max"]
    }, replace=True)
    print(new_transcription)

- Вывод:


    Было:
    mill_bd1_1_temp_l1_i103.0_h0.0-3.0_ni.min
    Стало:
    mill_bd1_1_temp_l1_i103.0_t0.0-3.0_ni.max


### Array Class

`Array` - представляет собой структуру данных для хранения массива значений сигнала источника сопоставленного с атрибутами
идентификатора сигнала источника в виде объекта `Transcription`. Он обеспечивает доступ к значениям массива сигнала, а также
предоставляет методы для замены значений и добавления атрибутов к идентификатору сигнала источника.

Класс принимает следующие аргументы:

- `transcription` (Transcription): Объект класса `Transcription`, содержащий информацию об атрибутах идентификатора сигнала. 
- `values` (numpy.array): Массив значений сигнала.
- `is_numeric` (bool): Флаг, указывающий, принадлежат ли типы значений в массиве следующим из доступных:
`AVAILABLE_TYPES` = (np.float, np.int, np.floating, np.integer)
- `is_billet` (bool): Флаг, указывающий, является ли массив значений координатами по длине заготовки.

При инициализации вычисляется ключ `key`, который используется для идентификации массива данных сигнала, 
и равен идентификатору сигнала.

    class Array:
    
        def __init__(
            self,
            transcription: Transcription,
            values: np.array,
            is_numeric: bool,
            is_billet: bool,
        ):
            self.transcription = transcription
            self.is_numeric = is_numeric
            self.is_billet = is_billet
            self.values = values
            self.key = str(self.transcription)
    
#### Метод `__repr__`
Метод `__repr__` возвращает строковое представление объекта класса `Array`.

        def __repr__(self):
            return (
                f"{str(self.transcription)}; "
                f"is_numeric={self.is_numeric}"
                f", is_billet={self.is_billet}"
            )
    
#### Метод `replace_values`
Метод `replace_values` заменяет значения массива новыми значениями `new_values`.

        def replace_values(self, new_values: np.array):
            self.values = new_values
    
#### Метод `append_keys_to_transcription`
Метод `append_keys_to_transcription` добавляет атрибуты или заменяет на новые к идентификатору массива данных сигнала.

        def append_keys_to_transcription(
            self, tags: Dict[str, List[str]], replace: bool = False
        ):
            new_transcription = self.transcription.add_tags(tags, replace)
            new_array = copy(self)
            new_array.transcription = new_transcription
            new_array.key = str(new_transcription)
            return new_array

- Пример без замены (`replace=False`):


    tags = {"aggregation": ["max"]}
    new_array = array.append_keys_to_transcription(tags, replace=False)
    print(new_array.key)

- Вывод:


    Было:
    mill_bd1_1_temp_l1_i103.0_h0.0-3.0_ni.min
    Стало:
    mill_bd1_1_temp_l1_i103.0_h0.0-3.0_ni.min_max

- Пример с заменой (`replace=True`):


    tags = {"aggregation": ["max"]}
    new_array = array.append_keys_to_transcription(tags, replace=True)
    print(new_array.key)

- Вывод:


    Было:
    mill_bd1_1_temp_l1_i103.0_h0.0-3.0_ni.min
    Стало:
    mill_bd1_1_temp_l1_i103.0_h0.0-3.0_ni.max


#### Зависимые классы

- `Transcription` Class: Расположен в модуле `naming`.


### FeaturesGenerator Class

`FeaturesGenerator` - это класс, который выполняет различные преобразования над массивами данных,
такие как вычисление абсолютных значений или их нормализация. Кроме того, этот класс обеспечивает уникальное
идентификационное имя для каждого массива данных путем изменения ключа транскрипции `transcription key`.

При создании объекта класса `FeaturesGenerator` инициализируется словарь `methods`, который содержит ключи,
соответствующие методам обработки данных.

    class FeaturesGenerator:
    
        def __init__(self):
            self.methods = {
                "abs": self.absolute,
                "norm": self.norm,
            }

#### Метод `get_method_values`
Метод `get_method_values` возвращает значения признаков на основе указанного метода.

        def get_method_values(self, method, *args, **kwargs):
            return self.methods[method](*args, **kwargs)

#### Метод `absolute`
Метод `absolute` вычисляет абсолютное значение для данных массива.


        @staticmethod
        def absolute(data_array: Array, *args, **kwargs) -> Array:
            """Модуль"""
            transcription = copy(data_array.transcription)
            transcription.add_tags({"preprocessing": ["abs"]}, )
            array = Array(
                transcription=transcription,
                values=abs(data_array.values),
                is_numeric=True,
                is_billet=False
            )
            return array

- Пример:


    features_generator = FeaturesGenerator()

    values = np.array([-559.94, 572.96, -582.03, 594.40, -606.77, 620.23, -640.36, 1023.74, -1062.85])
    array = Array(
        transcription=transcription,
        values=values,
        is_numeric=True,
        is_billet=False
    )

    result_array = features_generator.absolute(array) 
    result_array.values

- Вывод:


    Было:
    [-559.94, 572.96, -582.03, 594.40, -606.77, 620.23, -640.36, 1023.74, -1062.85]
    Стало:
    [559.94, 572.96, 582.03, 594.4, 606.77, 620.23, 640.36, 1023.74, 1062.85]

#### Метод `norm`
Метод `norm` выполняет нормализацию данных массива по заданным минимальным и максимальным значениям.


        @staticmethod
        def norm(data_array: Array, MIN, MAX, *args, **kwargs) -> Array:
            """Модуль"""
            transcription = copy(data_array.transcription)
            name = (
                data_array.transcription.workunit + "_"
                + data_array.transcription.name
            )
            transcription.add_tags({"preprocessing": ["norm"]})
            array = Array(
                transcription=transcription,
                values=np.array(
                    [
                        (val - MIN[name]) / (MAX[name] - MIN[name])
                        for val in data_array.values
                    ]
                ),
                is_numeric=True,
                is_billet=False
            )
            return array

- Пример:


    features_generator = FeaturesGenerator()

    values = np.array([-559.94, 572.96, -582.03, 594.40, -606.77, 620.23, -640.36, 1023.74, -1062.85])
    array = Array(
        transcription=transcription,
        values=values,
        is_numeric=True,
        is_billet=False
    )

    MIN = {"bd1_temp": -1062.85}
    MAX = {"bd1_temp": 1023.74}

    result_array = features_generator.norm(array, MIN, MAX)
    result_array.values

- Вывод:


    Было:
    [-559.94, 572.96, -582.03, 594.40, -606.77, 620.23, -640.36, 1023.74, -1062.85]
    Стало:
    [0.24102004, 0.78396331, 0.23043339, 0.79423845, 0.21857672, 0.8066175, 0.20247869, 1., 0.]


#### Зависимые классы:

- Методы класса `FeaturesGenerator` принимают на вход объекты класса `Array` Class: Расположен в модуле `source_data`.

---

Далее рассмотрим класс `SourceDataset`, так как класс `SecondaryFunctions` принимает объекты именно этого класса.

### SourceDataset Class

Объект класса `SourceDataset` представляет собой словарь массивов данных `Array`.

Класс `SourceDataset` представляет собой надстройку над классами `Array` и `Transcription`, который обеспечивает удобное
управление массивами данных, собранными из различных источников. Он позволяет создавать объекты, которые хранят словарь
массивов данных `Array`, где каждый массив представляет собой отдельный сигнал. Эти сигналы объединяются по определенному
источнику данных, например, `LNK100`, и хранятся в виде словаря `data`.

При инициализации объект класса `SourceDataset` принимает в качестве атрибута объект типа `SourceSettings`, структура которого 
определяется в модуле `dto`:

#### Структура `SourceSettings` из модуля `dto`

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

Структура `SourceSettings` содержит cледующие настроечные параметры:

- `source`: Имя источника данных, фичи (LNK100; U0; BD1; WBF_PIRO).
- `type`: Тип источника данных, фичи (target; feature).
- `handler`: Обработчик данных (BASE; WBFPIRO; WBFSINGLE).
- `rolling_number`: Номер прокатки.
- `encoding`: Кодировка данных (UTF-8; ANSI).
- `is_single`: Флаг, указывающий, что параметры сигнала имеют одно значение или более.
- `main_folder`: Корневая папка, с которой стартует рекурсивный поиск.
- `key_folder`: Ключевая папка источника данных.
- `nested_folders`: Вложенная папка внутри key_folder.
- `filename_key`: Ключ имени файла.
- `interpolation_type`: Тип интерполяции (`by_value`; `by_source`).
- `interpolation`: Значение интерполяции (необязательно).
- `segments`: Сегменты на которые разбиватеся заготовка.
- `forbidden_columns`: Имена колонок которые можно выкинуть из источника.
- `filtration_methods`: Методы фильтрации.
- `secondary_features`: вспомогательные функции для обработки и анализа данных.
- `aggregation_methods`: Методы аггрегации данных внутри сегмента.
- `billet_column`: Имя колонки с Billet points.
- `convert_columns`: Словарь для преобразования столбцов.

Имя ключа заготовки `self.billet_key` собирается с помощью класса `Transcription`, уже из полученного объекта `settings`.

    class SourceDataset

      def __init__(self, settings: SourceSettings):
          self.settings = settings
          self.data = {}
          self.billet_key = str(
              Transcription(
                  workunit=self.settings.source,
                  rolling_number=self.settings.rolling_number,
                  name=self.settings.billet_column,
                  interpolation="ni"
              )
          )

Шаблон формирования идентификатора ключа заготовки `self.billet_key`:

    {workunit}_{rolling_number}_{name}_{interpolation}

Пример:

    'BD1_1_billet_points_ni'

- Пример инициализации объекта класса `SourceDataset`:


    # Создание объекта SourceSettings
    settings = SourceSettings(
        source="LNK100",
        type="target",
        handler="BASE",
        rolling_number="",
        encoding="UTF-8",
        is_single=False,
        main_folder="\\ZSMK-9684-001\Data\2023",
        key_folder="ipsrnk",
        nested_folders="rail_points_files",
        filename_key="",
        interpolation_type="by_value",
        interpolation=103,
        segments={
                "LNK100_1": Segment(start_point=0, end_point=3, target_segments=[]),
                "LNK100_2": Segment(start_point=3, end_point=6, target_segments=["LNK100_1"]),
                "LNK100_3": Segment(start_point=30, end_point=70, target_segments=["LNK100_1"]),
                "LNK100_4": Segment(start_point=-3, end_point="end", target_segments=["LNK100_1"]),
                "LNK100_5": Segment(start_point=-6, end_point=-3, target_segments=["LNK100_1"])
            },
        forbidden_columns=[],
        filtration_methods=["std"],
        secondary_features=["abs"],
        aggregation_methods=['tg', 'max', 'min', 'median'],
        billet_column="BilletPoints",
        convert_columns={}
    )
    # Создание объекта SourceDataset
    dataset = SourceDataset(settings)

#### Метод `__iter__`
Метод `__iter__` используется для определения итератора, который позволяет проходиться по объекту `SourceSettings` с помощью цикла `for`.

        def __iter__(self):
            self._data_keys = list(self.data.keys())
            self._index = -1
            return self
    
#### Метод `__next__`
Метод `__next__` возвращает следующий элемент при итерации через объект `SourceSettings`.

        def __next__(self):
            if len(self._data_keys) != len(list(self.data.keys())):
                raise RuntimeError("object changed size during iterations")
            if self._index < len(self.data) - 1:
                self._index += 1
                return self.data[self._data_keys[self._index]]
            else:
                raise StopIteration
    
#### Метод `__getitem__`
Метод `__getitem__` позволяет получать доступ к элементам объекта `SourceSettings` по ключу, аналогично доступу к элементам словаря.

        def __getitem__(self, key: str):
            return self.data[key]

#### Метод `__repr__`
Метод `__repr__` возвращает строковое представление объекта.

        def __repr__(self):
            return f"{self.settings.source}; " \
                   f"billet='{self.settings.billet_column}'"

#### Метод `billet_array`
Метод `billet_array` возвращает массив данных `Array` (колонку) по координатам заготовки.

        def billet_array(self):
            """Возвращает Array, для которого is_billet = True"""
            return self.data[self.billet_key]

- Пример:


    dataset.billet_array().values

    Выведет колонку по координатам заготовки:
    array([0.025, 0.05 , 0.075, 0.1  , 0.125, 0.15 , 0.175, 0.2  , 0.225,
           0.25 , 0.275, 0.3  , 0.325, 0.35 , 0.375, 0.4  , 0.425, 0.45 ,
           0.475, 0.5  , 0.525, 0.55 , 0.575, 0.6  , 0.625, 0.65 , 0.675,
           0.7  , 0.725, 0.75 , 0.775, 0.8  , 0.825, 0.85 , 0.875, 0.9  ,
           0.925, 0.95 , 0.975, 1.   ...])

#### Метод `append_dataframe_to_source_data`
Метод `append_dataframe_to_source_data` принимает объект `pd.DataFrame` и преобразует его в набор данных в формате словаря
`Dict[Array]`, где ключом является строковое представление объекта `Transcription`, а значением - объект `Array`.


        def append_dataframe_to_source_data(self, data: pd.DataFrame):
            """Формирует из DataFrame набор данных в формате Dict[Array]"""
            for name, values in data.iteritems():
                values = values.to_numpy()
                is_numeric = self._is_numeric(values)
                transcription = Transcription(
                    workunit=self.settings.source,
                    rolling_number=self.settings.rolling_number,
                    name=str(name),
                    interpolation="ni"
                )
                is_billet = (
                    True if str(transcription) == self.billet_key else False
                )
                array = Array(
                    transcription=transcription,
                    values=values,
                    is_numeric=is_numeric,
                    is_billet=is_billet
                )
                self.data[str(transcription)] = array
    

- Пример преобразования `DataFrame` в набор данных в формате словаря `Dict[Array]`:


    df = pd.DataFrame({
        'BilletPoints': [0.025, 0.05, 0.075, 0.1, 0.125, ...],
        'Vert1000': [133, 133, 133, 133, 133, ...],
        'Vert1500': [224, 224, 224, 224, 224, ...],
        'Vert2000': [312, 312, 312, 312, 312, ...],
        'Vert3000': [400, 400, 400, 400, 400, ...],
        'Hor1000': [318, 318, 318, 318, 318, ...],
        'Hor1500': [747, 747, 747, 747, 747, ...],
        'Hor2000': [1152, 1152, 1152, 1152, 1152, ...],
        'Torsion': [-698, -698, -698, -698, -698, ...]
    })
    dataset.append_dataframe_to_source_data(df)
    display(dataset.data)

- Вывод:

    
    Теперь словарь data содержит объекты типа Array, где ключом является шаблон типа 
    {workunit}_{rolling_number}_{имя столбца DataFrame}_{interpolation="ni"}:

    {'LNK100_BilletPoints_ni': LNK100_BilletPoints_ni; is_numeric=True, is_billet=True,
     'LNK100_Vert1000_ni': LNK100_Vert1000_ni; is_numeric=True, is_billet=False,
     'LNK100_Vert1500_ni': LNK100_Vert1500_ni; is_numeric=True, is_billet=False,
     'LNK100_Vert2000_ni': LNK100_Vert2000_ni; is_numeric=True, is_billet=False,
     'LNK100_Vert3000_ni': LNK100_Vert3000_ni; is_numeric=True, is_billet=False,
     'LNK100_Hor1000_ni': LNK100_Hor1000_ni; is_numeric=True, is_billet=False,
     'LNK100_Hor1500_ni': LNK100_Hor1500_ni; is_numeric=True, is_billet=False,
     'LNK100_Hor2000_ni': LNK100_Hor2000_ni; is_numeric=True, is_billet=False,
     'LNK100_Torsion_ni': LNK100_Torsion_ni; is_numeric=True, is_billet=False}

- Значения массивов данных из словаря `data` можно вернуть методом `values` объекта `Array`:


    Например, выражение:

      dataset.data["LNK100_Vert2000_ni"].values

    Вернет массив данных:

      array([312, 312, 312, 312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
             312, 312, 312, 312, 312, 312, 312, 312, 311, 310, 309, 306, 308,
             305, 301, 297, 292, 289, 284, 278, 277, 275, 273, 271, 270, 268,
             264], dtype=int64)

#### Метод `append_array_to_source_data`
Метод `append_array_to_source_data` добавляет новый массив данных в виде объекта `Array` в словарь `data`
класса `SourceDataset`, где ключом является идентификатор сигнала. При этом, если `old_key` задан и равен ключу
заготовки `self.billet_key`, то ключ заготовки `self.billet_key` обновляется на идентификатор сигнала `Array`.

        def append_array_to_source_data(self, array: Array, old_key: str = None):
            """Добавляет новый Array"""
            self.data[array.key] = array
            if old_key == self.billet_key:
                self.billet_key = str(array.transcription)
    
#### Метод `replace_array_values`
Метод `replace_array_values` заменяет значения определенного массива данных `Array` в словаре `data`
по ключу идентификатора сигнала `Array` новыми значениями, переданными в качестве аргумента `new_values`.

        def replace_array_values(self, key: str, new_values: np.array):
            """Изменяет значения определнного Array"""
            self.data[key].replace_values(new_values)

#### Метод `remove_arrays`
Метод `remove_arrays` удаляет массивы из словаря `data` по ключам, переданным в виде списка `arrays`
 
        def remove_arrays(self, arrays: List[str]):
            for key in arrays:
                self.data.pop(key)
    
#### Метод `append_tags_to_array`
Метод `append_tags_to_array` добавляет атрибуты или заменяет на новые к идентификатору массива данных сигнала по указанному
ключу массива в словаре `self.data`.

        def append_tags_to_array(
            self, key: str, tags: Dict[str, List[str]], replace: bool = False
        ):
            new_array = self.data[key].append_keys_to_transcription(tags, replace)
            self.remove_arrays([key])
            self.append_array_to_source_data(new_array, key)
    
- Пример замены тегов к идентификаторам массивов данных:
  

    Было (имя ключа Array):
    BD1_1_temp.extrapolated_ni

  - Заменим теги у идентификатора массива:


      tags = {
          "interpolation": ["[i_0.0_103.0]"],
          "rolling_number": ["2"]
      }
      
      # Вызываем метод append_tags_to_array, передавая ключ массива и словарь тегов
      dataset.append_tags_to_array("BD1_1_temp.extrapolated_ni", tags, replace=True)

- Вывод:


    Стало (новое имя ключа Array):
    BD1_2_temp.extrapolated_[i_0.0_103.0]

#### Метод `append_tags_to_all_arrays`
Метод `append_tags_to_all_arrays` добавляет атрибуты или заменяет на новые ко всем идентификаторам (транскрипциям) массивов данных.

        def append_tags_to_all_arrays(
            self, tags: Dict[str, List[str]], replace: bool = False
        ):
            init_keys = self.data.keys()
            for key in init_keys:
                new_array = self.data[key].append_keys_to_transcription(
                    tags, replace
                )
                self.remove_arrays([key])
                self.append_array_to_source_data(new_array, key)

#### Метод `return_arrays_by_tags`
Метод `return_arrays_by_tags` возвращает массивы данных по заданным тегам, которые проверяются на наличие в транскрипции массива.

        def return_arrays_by_tags(self, tags: Dict[str, str]) -> List[Array]:
            arrays = []
            for array in self:
                for tag_key, tag_value in tags.items():
                    if getattr(array.transcription, tag_key) != tag_value:
                        break
                    arrays.append(array)
            return arrays
    
- Пример:


    Имеем следующий словарь массивов данных:

    {'LNK100_BilletPoints_ni': LNK100_BilletPoints_ni; is_numeric=True, is_billet=True,
     'LNK100_Vert1500_ni': LNK100_Vert1500_ni; is_numeric=True, is_billet=False,
     'LNK100_Vert2000_ni': LNK100_Vert2000_ni; is_numeric=True, is_billet=False,
     'LNK100_Vert3000_ni': LNK100_Vert3000_ni; is_numeric=True, is_billet=False,
     'LNK100_Hor1000_ni': LNK100_Hor1000_ni; is_numeric=True, is_billet=False,
     'LNK100_Hor1500_ni': LNK100_Hor1500_ni; is_numeric=True, is_billet=False,
     'LNK100_Hor2000_ni': LNK100_Hor2000_ni; is_numeric=True, is_billet=False,
     'LNK100_Torsion_ni': LNK100_Torsion_ni; is_numeric=True, is_billet=False,
     'LNK100_Vert1000_[i_0.0_103.0]': LNK100_Vert1000_[i_0.0_103.0]; is_numeric=True, is_billet=False}

- При этом хотим найти все массивы данных, которые производят интерполяцию к 103 метрам загототовки:


    # Cловарь с тегами, которые хотим проверить
    tags_to_check = {
        "interpolation": "[i_0.0_103.0]",
    }
    
    # Получаем список массивов, удовлетворяющих заданным тегам
    arrays_with_tags = dataset.return_arrays_by_tags(tags_to_check)

- Тогда получим следующий массив, удовлетворяющий условию фильтрации по тегам:


    LNK100_Vert1000_[i_0.0_103.0]': LNK100_Vert1000_[i_0.0_103.0]; is_numeric=True, is_billet=False
      
#### Метод `_is_numeric`
Метод `_is_numeric` проверяет принадлежат ли типы значений в массиве следующим из доступных:
`AVAILABLE_TYPES` = (np.float, np.int, np.floating, np.integer), которые указаны в модуле `constants`:


        @staticmethod
        def _is_numeric(values: np.array) -> bool:
            column_type = values.dtype
            is_numeric = False
            for parent_type in constants.AVAILABLE_TYPES:
                if np.issubdtype(column_type, parent_type):
                    is_numeric = True
                    break
            return is_numeric


### SecondaryFunctions Class

`SecondaryFunctions` - это класс, содержащий различные вспомогательные функции для обработки и анализа данных массивов
сигналов датасета `SourceDataset` источника данных.

    class SecondaryFunctions:
    
        def __init__(self):
            pass

#### Метод `sort_dataset_ascending_by_billet`
Метод `sort_dataset_ascending_by_billet` выполняет сортировку массивов данных объекта `SourceDataset` по возрастанию значений
столбца `billet points`. Для этого метод извлекает массив значений `billet points`, сортирует индексы значений по возрастанию
и применяет полученную последовательность индексов ко всем массивам данных в объекте `SourceDataset`. После сортировки
значения во всех массивах данных обновляются в соответствии с полученной последовательностью.

        @staticmethod
        def sort_dataset_ascending_by_billet(
            source_data: SourceDataset
        ) -> SourceDataset:
            """Сортировка billet points по возрастанию"""
            new_sequence = sorted(
                enumerate(source_data.billet_array().values), key=lambda x: x[1]
            )
            new_sequence = [val[0] for val in new_sequence]
            for data_array in source_data:
                new_values = [data_array.values[index] for index in new_sequence]
                source_data.append_tags_to_array(
                    data_array.key, {"secondary_functions": ["sortas"]}
                )
                source_data.replace_array_values(
                    data_array.key, np.array(new_values)
                )
            return source_data

#### Метод `approximate_billet_by_bounds`
Метод `approximate_billet_by_bounds` класса `SecondaryFunctions` производит аппроксимацию значений столбца `Billet`
датасета `SourceDataset` к заданной размерности. Для этого метод вычисляет верхнюю и нижнюю границы (`ub` и `lb`),
а также максимальное значение `Billet`. Затем метод применяет линейное масштабирование к значениям `Billet` таким образом,
чтобы они находились в интервале между нижней и верхней границей.

    new_value = (value * (ub - lb) / max_billet) + lb

Где:

- `new_value` - новое значение после масштабирования
- `value` - исходное значение
- `ub` - верхняя граница (верхнее ограничение); берется из настроек `settings` датасета по ключу `interpolation`.
- `lb` - нижняя граница (нижнее ограничение); равна 0
- `max_billet` - максимальное значение `Billet` в исходных данных.

Полученные значения заменяют исходные значения `Billet` в датасете `SourceDataset`. Теги `interpolation` с указанием
интервала аппроксимации добавляются ко всем транскрипциям массивов данных в датасета `SourceDataset`, например: 
- До апроксимации имя массива данных: `LNK100_Torsion_ni`
- После апроксимации к 103 метрам имя массива данных: `LNK100_Torsion_[i_0.0_103.0]`

        @staticmethod
        def approximate_billet_by_bounds(
            source_data: SourceDataset
        ) -> SourceDataset:
            """Аппроксимация столбца Billet к
            необходимой, строго фиксированной размерности"""
            ub = source_data.settings.interpolation
            lb = 0
            max_billet = max(source_data.billet_array().values)
            new_billet_values = (
                source_data.billet_array().values * (ub - lb) / max_billet
            ) + lb
            source_data.replace_array_values(
                source_data.billet_key, new_billet_values
            )
            source_data.append_tags_to_all_arrays(
                {"interpolation": [f"[i_{lb:.1f}_{ub:.1f}]"]}, replace=True
            )
            return source_data

#### Метод `append_workcenter_to_transcription`
Метод `append_workcenter_to_transcription` добавляет теги с информацией о рабочем блоке (`mill` (стан), `ccm` (МНЛЗ))
к транскрипции каждого массива данных в датасете `SourceDataset`.

        @staticmethod
        def append_workcenter_to_transcription(
            source_data: SourceDataset, workcenters: Dict[str, str]
        ) -> SourceDataset:
            for data_array in source_data:
                tags = workcenters[data_array.transcription.workunit]
                source_data.append_tags_to_array(
                    data_array.key, {"workcenter": [tags]}, replace=True
                )
            return source_data

#### Метод `cut_wbf_pirometer_signals`
Метод `cut_wbf_pirometer_signals` ищет точки разделения на основе сигнала и формирует сегменты согласно заданным параметрам окон.

#### Параметры:
- `source_data`: Объект `SourceDataset`, содержащий массивы данных и транскрипции.
- `piro_settings`: Словарь с настройками окон для нарезки сегментов.

#### Возвращаемое значение:
Объект `SourceDataset` с нарезанными сегментами.

Словарь с настройками для нарезки сегментов располагается в модуле `materials` в формате JSON и обозначен как
`wbf_piro_cutter_settings.json`:

      {
          "PIROMETERS": ["TEMP_PIR1", "TEMP_PIR2", "TEMP_PIR3"],
          "SETTINGS": 
          {
                  "BEAM_VERT_ENC": 
                  [
                      [-0.1, 0.1, 20, 0.5, 10, 5],
                      [0.5, 10, 5, -0.1, 0.1, 20],
                      [-0.1, 0.1, 20, -10, -1, 5],
                      [-10, -1, 5, -0.1, 0.1, 20],
                      [-0.1, 0.1, 20, 0.5, 10, 5]
                  ],
                  "BEAM_HOR_ENC_POS": 
                  [
                      [-0.1, 0.1, 20, 0.5, 10, 5],
                      [5, 50, 100, -0.1, 0.1, 20],
                      [-1, 1, 20, -10, -1, 5],
                      [-10, -1, 20, -0.1, 0.1, 20]
                  ],
                  "DM_ENC_HOR_POS": 
                  [	
                      [0, 2, 20, 50, 200, 20],
                      [5, 100, 10, 0, 2, 20],
                      [0.1, 1, 20, 5, 200, 5],
                      [2, 200, 20, 0, 0.1, 5]
                  ],
                  "DML_VERT_POS": 
                  [
                      [-100, -5, 5, -1, 1, 20],
                      [-1, 1, 20, 5, 100, 5],
                      [5, 100, 5, -1, 1, 20],
                      [-1, 1, 20, 5, 100, 5],
                      [5, 100, 5, -1, 1, 5],
                      [-1, 1, 20, -100, -5, 5]
                  ],
                  "DMR_VERT_POS": 
                  [
                      [-100, -5, 5, -1, 1, 20],
                      [-1, 1, 20, 5, 100, 5],
                      [5, 100, 5, -1, 1, 20],
                      [-1, 1, 20, 5, 100, 5],
                      [5, 100, 5, -1, 1, 5],
                      [-1, 1, 20, -100, -5, 5]
                  ]
          }
      }

В данном примере каждый параметр в `SETTINGS` имеет определенное кол-во настроечных параметров для окон,
где параметры указываются в виде списков. Эти параметры определяют, каким образом будет выполняться нарезка данных на сегменты.

Точки разделения задаются в формате левого окна `L_WIN`, правого окна `R_WIN` и допустимого изменения сигнала
`L_DELTA_MIN`, `L_DELTA_MAX`, `R_DELTA_MIN`, `R_DELTA_MAX`. Параметры для определения точек разделения задаются
структурой `CutPoint`, которая определена в модуле `dto`:

    @dataclass
    class CutPoint:
        L_DELTA_MIN: float # Минимальное изменение слева.
        L_DELTA_MAX: float # Максимальное изменение слева.
        L_WIN: int # Размер окна слева.
        R_DELTA_MIN: float # Минимальное изменение справа.
        R_DELTA_MAX: float # Максимальное изменение справа.
        R_WIN: int # Размер окна справа.

Эти параметры определяют условия, при которых точки данных считаются подходящими для выделения в сегменты.
Каждый новый лист может представлять собой новый набор параметров для нарезки данных на сегменты с различными характеристиками.


Описание метода `cut_wbf_pirometer_signals` приведено ниже:

        @staticmethod
        def cut_wbf_pirometer_signals(
            source_data: SourceDataset, piro_settings: Dict[str, Dict[str, str]]
        ):

- Получение всех целевых сегментов `target_segments` из настроек исходного набора данных:

            all_target_segments = list(
                set(
                    [
                        target for seg in source_data.settings.segments.values()
                        for target in seg.target_segments
                    ]
                )
            )

- `Нахождение точек разделения` осуществляется путем анализа сигнала и определения моментов, когда сигнал значительно
меняется в пределах заданных параметров. Здесь используются параметры, такие как `L_WIN`, `R_WIN`, `L_DELTA_MIN`,
`L_DELTA_MAX`, `R_DELTA_MIN`, `R_DELTA_MAX`, которые задают окно анализа и минимальные/максимальные изменения сигнала слева
и справа от текущей точки для определения точек разделения.


            all_cut_indexes = []
            for name, points in piro_settings["SETTINGS"].items():
                array = source_data.return_arrays_by_tags({"name": name})[0]
    
                # Ищем точки разделения
                cut_indexes = []
                point_idx = 0
                col_len = len(array.values)
                for idx in range(col_len):
                    point_data = CutPoint(*points[point_idx])
                    if (idx - point_data.L_WIN < 0
                            or idx + point_data.R_WIN >= col_len):
                        continue
                    l_delta = (
                        array.values[idx] - array.values[idx - point_data.L_WIN]
                    )
                    r_delta = (
                        array.values[idx + point_data.R_WIN] - array.values[idx]
                    )
                    if (point_data.L_DELTA_MIN <= l_delta <= point_data.L_DELTA_MAX
                            and (point_data.R_DELTA_MIN <= r_delta <=
                                 point_data.R_DELTA_MAX)):
                        cut_indexes.append(idx)
                        if point_idx + 1 < len(points):
                            point_idx += 1
                        else:
                            break
                all_cut_indexes.extend(cut_indexes)


- `Создание сегментов на основе точек разделения`. После того, как найдены точки разделения, то есть сформирован промежуточный или окончательный массив `all_cut_indexes`, 
который содержит индексы точек разделения, например:


    all_cut_indexes = [77, 100, 153, 469, 495, 752, 773, 1152, 1172, 1596, 1706, 1772, 1772, 1816, 1820, 1892, 2067, 2095, 2098, 2183, 2186, 2322, 2326]

мы создаем сегменты на основе этих точек. Каждая точка разделения определяет начало и конец сегмента, а также связанные сегменты, к которым этот сегмент относится.

Первые сегменты, такие как `BEAM_VERT_ENC_0`, `DML_VERT_POS_3` и т.д., создаются на основе точек разделения, найденных в процессе обработки данных
для сигналов типа: `BEAM_VERT_ENC`; `BEAM_HOR_ENC_POS`; `DM_ENC_HOR_POS`; `DML_VERT_POS`; `DMR_VERT_POS`. 
Каждый из этих сегментов имеет начальную и конечную точки, определенные временем. Например, `BEAM_VERT_ENC_0` начинается в момент времени `2024-03-07 23:59:33.107000`
и заканчивается в `2024-03-07 23:59:53.057000`.

                for number, indexes in enumerate(zip(cut_indexes[:-1],
                                                     cut_indexes[1:])):
                    source_data.settings.segments[f"{name}_{number}"] = Segment(
                        start_point=source_data.billet_array().values[indexes[0]],
                        end_point=source_data.billet_array().values[indexes[1]],
                        target_segments=all_target_segments
                    )

Пример сегментов, полученных в процессе обработки данных для сигналов типа: `BEAM_VERT_ENC`; `BEAM_HOR_ENC_POS`; `DM_ENC_HOR_POS`; `DML_VERT_POS`; `DMR_VERT_POS`:

     'BEAM_VERT_ENC_0': Segment(start_point='2024-03-07 23:59:33.107000', end_point='2024-03-07 23:59:53.057000', target_segments=['LNK100_1']),
     'BEAM_VERT_ENC_1': Segment(start_point='2024-03-07 23:59:53.057000', end_point='2024-03-08 00:00:08.907000', target_segments=['LNK100_1']),
     'BEAM_VERT_ENC_2': Segment(start_point='2024-03-08 00:00:08.907000', end_point='2024-03-08 00:00:30.957000', target_segments=['LNK100_1']),
     'BEAM_VERT_ENC_3': Segment(start_point='2024-03-08 00:00:30.957000', end_point='2024-03-08 00:01:04.107000', target_segments=['LNK100_1']),
     'BEAM_HOR_ENC_POS_0': Segment(start_point='2024-03-07 23:59:55.457000', end_point='2024-03-08 00:00:10.207000', target_segments=['LNK100_1']),
     'BEAM_HOR_ENC_POS_1': Segment(start_point='2024-03-08 00:00:10.207000', end_point='2024-03-08 00:00:32.607000', target_segments=['LNK100_1']),
     'BEAM_HOR_ENC_POS_2': Segment(start_point='2024-03-08 00:00:32.607000', end_point='2024-03-08 00:00:54.757000', target_segments=['LNK100_1']),
     'DM_ENC_HOR_POS_0': Segment(start_point='2024-03-08 00:01:14.507000', end_point='2024-03-08 00:01:23.407000', target_segments=['LNK100_1']),
     'DML_VERT_POS_0': Segment(start_point='2024-03-07 23:59:28.907000', end_point='2024-03-08 00:01:07.607000', target_segments=['LNK100_1']),
     'DML_VERT_POS_1': Segment(start_point='2024-03-08 00:01:07.607000', end_point='2024-03-08 00:01:09.907000', target_segments=['LNK100_1']),
     'DML_VERT_POS_2': Segment(start_point='2024-03-08 00:01:09.907000', end_point='2024-03-08 00:01:25.057000', target_segments=['LNK100_1']),
     'DML_VERT_POS_3': Segment(start_point='2024-03-08 00:01:25.057000', end_point='2024-03-08 00:01:29.307000', target_segments=['LNK100_1']),
     'DML_VERT_POS_4': Segment(start_point='2024-03-08 00:01:29.307000', end_point='2024-03-08 00:01:36.707000', target_segments=['LNK100_1']),
     'DMR_VERT_POS_0': Segment(start_point='2024-03-07 23:59:30.057000', end_point='2024-03-08 00:01:07.607000', target_segments=['LNK100_1']),
     'DMR_VERT_POS_1': Segment(start_point='2024-03-08 00:01:07.607000', end_point='2024-03-08 00:01:10.107000', target_segments=['LNK100_1']),
     'DMR_VERT_POS_2': Segment(start_point='2024-03-08 00:01:10.107000', end_point='2024-03-08 00:01:24.907000', target_segments=['LNK100_1']),
     'DMR_VERT_POS_3': Segment(start_point='2024-03-08 00:01:24.907000', end_point='2024-03-08 00:01:29.457000', target_segments=['LNK100_1']),
     'DMR_VERT_POS_4': Segment(start_point='2024-03-08 00:01:29.457000', end_point='2024-03-08 00:01:36.507000', target_segments=['LNK100_1']),

- В конце создаются сегменты `ALL_PTS_X` для охвата всех точек данных между соседними точками разделения.

            all_cut_indexes = sorted(all_cut_indexes)
            for number, indexes in enumerate(zip(all_cut_indexes[:-1],
                                                 all_cut_indexes[1:])):
                source_data.settings.segments[f"ALL_PTS_{number}"] = Segment(
                    start_point=source_data.billet_array().values[indexes[0]],
                    end_point=source_data.billet_array().values[indexes[1]],
                    target_segments=all_target_segments
                )
            return source_data
    
Например, `ALL_PTS_0` охватывает данные с `2024-03-07 23:59:28.907000` до `2024-03-07 23:59:30.057000`.

     'ALL_PTS_0': Segment(start_point='2024-03-07 23:59:28.907000', end_point='2024-03-07 23:59:30.057000', target_segments=['LNK100_1']),
     'ALL_PTS_1': Segment(start_point='2024-03-07 23:59:30.057000', end_point='2024-03-07 23:59:33.107000', target_segments=['LNK100_1']),
     'ALL_PTS_2': Segment(start_point='2024-03-07 23:59:33.107000', end_point='2024-03-07 23:59:53.057000', target_segments=['LNK100_1']),
     'ALL_PTS_3': Segment(start_point='2024-03-07 23:59:53.057000', end_point='2024-03-07 23:59:55.457000', target_segments=['LNK100_1']),
     'ALL_PTS_4': Segment(start_point='2024-03-07 23:59:55.457000', end_point='2024-03-08 00:00:08.907000', target_segments=['LNK100_1']),
     'ALL_PTS_5': Segment(start_point='2024-03-08 00:00:08.907000', end_point='2024-03-08 00:00:10.207000', target_segments=['LNK100_1']),
     'ALL_PTS_6': Segment(start_point='2024-03-08 00:00:10.207000', end_point='2024-03-08 00:00:30.957000', target_segments=['LNK100_1']),
     'ALL_PTS_7': Segment(start_point='2024-03-08 00:00:30.957000', end_point='2024-03-08 00:00:32.607000', target_segments=['LNK100_1']),
     'ALL_PTS_8': Segment(start_point='2024-03-08 00:00:32.607000', end_point='2024-03-08 00:00:54.757000', target_segments=['LNK100_1']),
     'ALL_PTS_9': Segment(start_point='2024-03-08 00:00:54.757000', end_point='2024-03-08 00:01:04.107000', target_segments=['LNK100_1']),
     'ALL_PTS_10': Segment(start_point='2024-03-08 00:01:04.107000', end_point='2024-03-08 00:01:07.607000', target_segments=['LNK100_1']),
     'ALL_PTS_11': Segment(start_point='2024-03-08 00:01:07.607000', end_point='2024-03-08 00:01:07.607000', target_segments=['LNK100_1']),
     'ALL_PTS_12': Segment(start_point='2024-03-08 00:01:07.607000', end_point='2024-03-08 00:01:09.907000', target_segments=['LNK100_1']),
     'ALL_PTS_13': Segment(start_point='2024-03-08 00:01:09.907000', end_point='2024-03-08 00:01:10.107000', target_segments=['LNK100_1']),
     'ALL_PTS_14': Segment(start_point='2024-03-08 00:01:10.107000', end_point='2024-03-08 00:01:14.507000', target_segments=['LNK100_1']),
     'ALL_PTS_15': Segment(start_point='2024-03-08 00:01:14.507000', end_point='2024-03-08 00:01:23.407000', target_segments=['LNK100_1']),
     'ALL_PTS_16': Segment(start_point='2024-03-08 00:01:23.407000', end_point='2024-03-08 00:01:24.907000', target_segments=['LNK100_1']),
     'ALL_PTS_17': Segment(start_point='2024-03-08 00:01:24.907000', end_point='2024-03-08 00:01:25.057000', target_segments=['LNK100_1']),
     'ALL_PTS_18': Segment(start_point='2024-03-08 00:01:25.057000', end_point='2024-03-08 00:01:29.307000', target_segments=['LNK100_1']),
     'ALL_PTS_19': Segment(start_point='2024-03-08 00:01:29.307000', end_point='2024-03-08 00:01:29.457000', target_segments=['LNK100_1']),
     'ALL_PTS_20': Segment(start_point='2024-03-08 00:01:29.457000', end_point='2024-03-08 00:01:36.507000', target_segments=['LNK100_1']),
     'ALL_PTS_21': Segment(start_point='2024-03-08 00:01:36.507000', end_point='2024-03-08 00:01:36.707000', target_segments=['LNK100_1'])


Пример нахождения точек разделения:

Допустим, у нас есть массив сигнала пирометра:

    [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

И следующие настройки для точек разделения:

    L_WIN = 2
    R_WIN = 1
    L_DELTA_MIN = 5
    L_DELTA_MAX = 15
    R_DELTA_MIN = 5
    R_DELTA_MAX = 15

Начинаем с первой точки и проверяем каждую точку по очереди.

1. Для первой точки 10 мы проверяем, есть ли у нас достаточное количество точек слева (L_WIN = 2) и справа (R_WIN = 1).
Если да, мы вычисляем изменение сигнала слева и справа.
2. Для точек 10 и 15 у нас нет достаточного количества точек слева, поэтому мы переходим к точке 20.
3. Для точки 20, у нас есть достаточное количество точек слева и справа. Мы вычисляем изменение сигнала слева и справа.
4. Изменение слева и справа вычисляется следующим образом:

    
    L_DELTA = value[idx] - value[idx - L_WIN]

    R_DELTA = value[idx + R_WIN] - value[idx]

где

 - idx: индекс текущего значения;
 - L_WIN: необходимое кол-во точек слева;
 - R_WIN: необходимое кол-во точек справа.

Тогда изменение слева составит: L_DELTA = 20 - 10 = 10;
а изменение справа: R_DELTA = 25 - 20 = 5.

5. Если изменение сигнала слева и справа находится в заданных пределах (5 <= L_DELTA <= 15 и 5 <= R_DELTA <= 15), мы добавляем эту точку в список точек разделения.
Этот процесс продолжается до конца массива.

Пример создания сегментов:
Предположим, что мы нашли следующие точки разделения: [2, 5, 8, 11]. Это означает, что у нас есть три сегмента:

- Сегмент 1: от точки 2 до точки 5
- Сегмент 2: от точки 5 до точки 8
- Сегмент 3: от точки 8 до точки 11

#### Метод `convert_date_columns_to_numeric`
Метод `convert_date_columns_to_numeric` преобразует столбцы с датами в числовой формат, заменяя значения дат на количество
секунд относительно базовой даты.


        @staticmethod
        def convert_date_columns_to_numeric(
            source_data: SourceDataset, tags_list: List[Dict[str, str]]
        ) -> SourceDataset:
            for tag in tags_list:
                tag_arrays = source_data.return_arrays_by_tags(tag)
                for array in tag_arrays:
                    date_values = [
                        (pd.Timestamp(value) - constants.BASE_TIME).total_seconds()
                        for value in array.values
                    ]
                    min_data = min(date_values)
                    converted_values = [value - min_data for value in date_values]
                    source_data.replace_array_values(
                        array.key, np.array(converted_values)
                    )
                    source_data.append_tags_to_array(
                        array.key, {"secondary_functions": ["converted"]}
                    )
            return source_data

Пример:

До преобразования:

    ['2024-03-07T23:59:24.507000000', '2024-03-07T23:59:24.557000000', '2024-03-07T23:59:24.607000000', ...,
     '2024-03-08T00:01:38.907000000', '2024-03-08T00:01:38.957000000', '2024-03-08T00:01:39.007000000']

После преобразования:
    
    [0.00000000e+00, 5.00000715e-02, 1.00000024e-01, ..., 1.34400000e+02, 1.34450000e+02, 1.34500000e+02]


#### Зависимые классы

- Методы класса `SecondaryFunctions` принимают на вход объекты классов `SourceDataset` Class: Расположен в модуле `source_data`.

---

Ниже рассмотрены классы модуля сегментного агрегатора `segments_agregator`.

### AggregatedValue Class

Класс `AggregatedValue` представляет собой объект, содержащий информацию о полученном агрегированном значении
массива данных сегмента, которое было получено одним из следующих методов агрегации: [median, max, min, mean, tg].

Параметры:
`segment_id`: строка, идентификатор сегмента.
`transcription`: объект типа `Transcription`, содержащий информацию об идентификаторе массива данных сегмента (mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_median).
`value`: агрегированное значение, полученное по сегменту массива данных.
`is_bad`: флаг, указывающий, является ли значение недопустимым (по умолчанию False).
`bad_reason`: строка, содержащая причину, по которой значение считается недопустимым (по умолчанию None).

    class AggregatedValue:
    
        def __init__(
            self,
            segment_id: str,
            transcription: Transcription,
            value: Optional[float],
            is_bad: bool = False,
            bad_reason: str = None
        ):
            self.segment_id = segment_id
            self.transcription = transcription
            self.value = value
            self.is_bad = is_bad
            self.bad_reason = bad_reason
    
Метод `__repr__` возвращает строковое представление объекта `AggregatedValue`, отображая информацию об
идентификаторе (транскрипции) массива данных и агрегированное значение массива данных сегмента.

        def __repr__(self):
            return f"{str(self.transcription)}, value={self.value}"

Общий вид объекта `AggregatedValue`:

    agg_value_seg = AggregatedValue(
                    segment_id="LNK100_1",
                    transcription="mill_LNK100_1_Vert1500_[i_0.0_103.0]_0_3_max",
                    value=25.0,
                    is_bad=False,
                    bad_reason=None
                )

    repr(agg_value_seg) => {mill_LNK100_1_Vert1500_[i_0.0_103.0]_0_3_max, value=25.0}

### AggregatedSegment Class

Класс `AggregatedSegment` позволяет создавать объекты, которые хранят информацию по сегменту массива данных,
и все его агрегированные значения в виде словаря `Dict[str, AggregatedValue]`.

Параметры:
`name`: строка, имя агрегированного сегмента.
`start_point`: начальная точка сегмента.
`end_point`: конечная точка сегмента.
`values`: словарь, содержащий агрегированные значения сегмента, где ключ - идентификатор агрегированного значения
массива данных сегмента, а значение - объект типа `AggregatedValue`.

    class AggregatedSegment(dict):
    
        def __init__(
            self,
            name: str,
            start_point: float,
            end_point: float,
            values: Dict[str, AggregatedValue] = ()
        ):
            self._name = name
            self._start_point = start_point
            self._end_point = end_point
            super(AggregatedSegment, self).__init__(values)
    
#### Метод `append_value`
Метод `append_value` принимает список объектов `AggregatedValue` и добавляет их в словарь агрегированных значений сегмента массива данных.

        def append_value(self, values: List[AggregatedValue]):
            for value in values:
                self[str(value.transcription)] = value
    
#### Метод `name`
Метод `name` возвращает имя агрегированного сегмента.

        def name(self):
            return self._name
    
#### Метод `start_point`
Метод `start_point` возвращает начальную точку сегмента.

        def start_point(self):
            return self._start_point
    
#### Метод `end_point`
Метод `start_point` возвращает конечную точку сегмента.

        def end_point(self):
            return self._end_point

Общий вид объекта `AggregatedSegment`:

    segment = AggregatedSegment(
        name="LNK100_1",
        start_point=0,
        end_point=3,
        values={
            "mill_LNK100_1_Vert2000_[i_0.0_103.0]_0_3_min": AggregatedValue(segment_id="LNK100_1", transcription="mill_LNK100_1_Vert2000_[i_0.0_103.0]_0_3_median", value=15.0),
            "mill_LNK100_1_Vert2000_[i_0.0_103.0]_0_3_median", transcription="mill_LNK100_1_Vert2000_[i_0.0_103.0]_0_3_min", value=20.0),
            # Другие агрегированные сегменты...
        }
    )

    segment =>
    {'mill_LNK100_1_Vert2000_[i_0.0_103.0]_0_3_min': mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_min, value=15.0,
    'mill_LNK100_1_Vert2000_[i_0.0_103.0]_0_3_median': mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_median, value=20.0,
    ...}

### AggregatedSourceDict Class

Объект класса `AggregatedSourceDict` представляет собой словарь сегментов и содержит общую информацию о настройках и
источнике данных.

    class AggregatedSourceDict(dict):
    
        def __init__(
            self,
            settings: SourceSettings,
            segments: Dict[str, AggregatedSegment] = ()
        ):
            super(AggregatedSourceDict, self).__init__(segments)
            self._source = settings.source
            self._is_target = True if settings.type == 'target' else False
            self._settings = settings
    
#### Метод `__repr__`
Метод `__repr__` подсчитывает общее количество плохих агрегированных значений в сегментах массивов и общее количество
агрегированных значений по всем сегментам, а затем возвращает строку, содержащую эти данные.

        def __repr__(self):
            count_bad = sum(
                [
                    1 for seg_vals in self.values() for value in seg_vals.values()
                    if value.is_bad
                ]
            )
            count_values = sum(len(seg_vals) for seg_vals in self.values())
            return f"AggregatedSource(total_values={count_values}, " \
                   f"bad_values={count_bad})"
    
#### Метод `append_segment`
Метод `append_segment` добавляет агрегированный сегмент к словарю сегментов `AggregatedSourceDict` по имени сегмента.

        def append_segment(self, segment: AggregatedSegment):
            self[segment.name()] = segment
    
#### Метод `source`
Метод `source` возвращает имя источника данных, предоставленного в настройках `SourceSettings`.

        def source(self):
            return self._source
    
#### Метод `settings`
Метод `settings` возвращает возвращает метаданные `SourceSettings` по всему словарю сегментов `AggregatedSourceDict`.

        def settings(self):
            return self._settings
    
#### Метод `is_target`
Метод `is_target` определяет к какому типу относится массив данных параметра, по которому создан данный словарь сегментов,
к целевой переменной `target` или к фиче.

        def is_target(self):
            return self._is_target

### SegmentsAggregator Class

При инициализации объекта класса `SegmentsAggregator` создается словарь `methods`, который содержит различные методы
агрегации данных для сегментов. Ключами словаря являются строки, представляющие названия методов агрегации, а
значениями - соответствующие методы класса `SegmentsAggregator`.

    class SegmentsAggregator:
    
        def __init__(self):
            self.methods = {
                "median": self.median_aggregate,
                "min": self.min_aggregate,
                "max": self.max_aggregate,
                "mean": self.mean_aggregate,
                "tg": self.tg_aggregate,
            }

#### Метод `return_segment_values`

Метод принимает сегменты массивов данных, структура которых указана в модуле `dto`:

    @dataclass
    class Segment:
        start_point: float # Начальная точка
        end_point: float # Конечная точка 
        target_segments: List[str] # Имена сегментов target, к которым будет сопоставлен данный сегмент

Каждый сегмент имеет свой собственный идентификатор.
Пример сегментов по времени:

    'DML_VERT_POS_0': Segment(start_point='2024-03-07 23:59:28.907000', end_point='2024-03-08 00:01:07.607000', target_segments=['LNK100_1']),
    'DML_VERT_POS_1': Segment(start_point='2024-03-08 00:01:07.607000', end_point='2024-03-08 00:01:09.907000', target_segments=['LNK100_1'])

Пример сегментов по длине заготовки:

    "U0_1": Segment(start_point=0, end_point=3, target_segments=["LNK100_1"]),
    "U0_2": Segment(start_point=0, end_point="end", target_segments=["LNK100_1"])

Метод `return_segment_values` извлекает значения массива данных, которые соответствуют интервалу длины заготовки,
определенному точками `start_point` и `end_point` в сегменте. 

    @staticmethod
    def return_segment_values(
        segment: Segment, billet_array: Array, data_array: Array
    ) -> np.array:

Если начальная точка `start_point` или конечная точка `end_point` в сегменте отрицательные, то это означает, что 
берутся точки на заготовке за указанный интервал, начиная с хвоста заготовки, поэтому происходит перерасчет координаты.

        # Начальная точка
        if segment.start_point < 0:
            start_point = max(billet_array.values) + segment.start_point
        else:
            start_point = segment.start_point

        # Конечная точка
        if segment.end_point == "end":
            end_point = max(billet_array.values)
        elif segment.end_point < 0:
            end_point = max(billet_array.values) + segment.end_point
        else:
            end_point = segment.end_point

Метод `return_segment_values` возвращает:
- Массив `points`, содержащий все координаты или временные отметки, которые находятся в интервале от 0 до 3 в столбце `billet points`.
- Массив `segment_data`, содержащий значения массива данных, которые попадают в заданный интервал между `start_point` и `end_point`.

        # Сегментация
        points = billet_array.values[(billet_array.values < 3)
                                     & (billet_array.values > 0)]
        segment_data = [
            value for bil, value in zip(billet_array.values, data_array.values)
            if end_point >= bil >= start_point
        ]
        return np.array(points), np.array(segment_data)

Пример для сегмента 

    "LNK100_1": Segment(start_point=0, end_point=3, target_segments=[])

Получим следующие координаты точек `billet points` заготовки в интервале от 0 до 3:

    points = (array([0.025, 0.05 , 0.075, 0.1  , 0.125, 0.15 , 0.175, 0.2  , 0.225,
                      0.25 , 0.275, 0.3  , 0.325, 0.35 , 0.375, 0.4  , 0.425, 0.45 ,
                      0.475, 0.5  , 0.525, 0.55 , 0.575, 0.6  , 0.625, 0.65 , 0.675,
                      0.7  , 0.725, 0.75 , 0.775, 0.8  , 0.825, 0.85 , 0.875, 0.9  ,
                      0.925, 0.95 , 0.975, 1.   , 1.025, 1.05 , 1.075, 1.1  , 1.125,
                      1.15 , 1.175, 1.2  , 1.225, 1.25 , 1.275, 1.3  , 1.325, 1.35 ,
                      1.375, 1.4  , 1.425, 1.45 , 1.475, 1.5  , 1.525, 1.55 , 1.575,
                      1.6  , 1.625, 1.65 , 1.675, 1.7  , 1.725, 1.75 , 1.775, 1.8  ,
                      1.825, 1.85 , 1.875, 1.9  , 1.925, 1.95 , 1.975, 2.   , 2.025,
                      2.05 , 2.075, 2.1  , 2.125, 2.15 , 2.175, 2.2  , 2.225, 2.25 ,
                      2.275, 2.3  , 2.325, 2.35 , 2.375, 2.4  , 2.425, 2.45 , 2.475,
                      2.5  , 2.525, 2.55 , 2.575, 2.6  , 2.625, 2.65 , 2.675, 2.7  ,
                      2.725, 2.75 , 2.775, 2.8  , 2.825, 2.85 , 2.875, 2.9  , 2.925,
                      2.95 , 2.975])

И следующие значения массива данных, попавшие в заданный интервал:

    segment_data = array([181., 181., 181., 181., 181., 181., 181., 181., 181., 181., 181.,
                          181., 181., 181., 181., 181., 181., 181., 181., 181., 184., 187.,
                          189., 191., 192., 193., 195., 195., 194., 194., 194., 194., 194.,
                          194., 191., 189., 185., 183., 179., 177., 175., 172., 171., 170.,
                          168., 168., 167., 167., 168., 167., 167., 167., 166., 163., 160.,
                          158., 155., 153., 148., 143., 139., 134., 131., 129., 126., 123.,
                          121., 119., 118., 118., 118., 118., 120., 121., 125., 128., 131.,
                          134., 135., 134., 133., 134., 133., 135., 134., 135., 136., 135.,
                          135., 136., 136., 135., 135., 136., 137., 135., 134., 133., 135.,
                          136., 136., 136., 137., 137., 137., 137., 135., 134., 133., 133.,
                          131., 131., 128., 126., 122., 118., 116., 114., 114., 115.])

#### Метод `get_method_value` 
Метод `get_method_value` необходим для получения агрегированного значения на основе указанного метода агрегации данных [median, max, min, mean, tg].

    def get_method_value(
        self,
        segment_id: str,
        transcription: Transcription,
        method: str,
        *args,
        **kwargs,
    ) -> AggregatedValue:
        return AggregatedValue(
            segment_id, transcription, self.methods[method](*args, **kwargs)
        )

#### Метод `median_aggregate` 
Метод `median_aggregate` вычисляет медиану значений в массиве данных и возвращает ее.

    @staticmethod
    def median_aggregate(data_array: np.array):
        return np.median(data_array)

#### Метод `max_aggregate` 
Метод `max_aggregate` находит максимальное значение в массиве данных и возвращает его.

    @staticmethod
    def max_aggregate(data_array: np.array):
        return np.max(data_array)

#### Метод `min_aggregate` 
Метод `min_aggregate` находит минимальное значение в массиве данных и возвращает его.

    @staticmethod
    def min_aggregate(data_array: np.array):
        return np.min(data_array)

#### Метод `mean_aggregate` 
Метод `mean_aggregate` вычисляет среднее значение массива данных и возвращает его.

    @staticmethod
    def mean_aggregate(data_array: np.array):
        return np.mean(data_array)

#### Метод `tg_aggregate` 
Метод `tg_aggregate` возвращает коэффициент наклона прямой, аппроксимирующей зависимость между данными массива 
и координатами заготовки.

    @staticmethod
    def tg_aggregate(data_array: np.array, points: np.array):
        extended_points = np.c_[points, np.ones(len(points))]
        tg, _ = np.linalg.lstsq(extended_points, data_array, rcond=None)[0]

        return tg

#### Метод `_rotate` 
Метод `_rotate` выполняет поворот графика данных вокруг начала координат на угол, вычисленный с использованием коэффициента
наклона прямой.

    def _rotate(self, data_array: np.array, points: np.array):
        angle = np.arctan(self.tg_aggregate(data_array, points))

        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle),
                                               np.cos(angle)]]
        )

        # Смещаем график к началу координат
        x_centered = points - np.mean(points)
        y_centered = data_array - np.mean(data_array)

        # Поворачиваем график
        xy_rotated = np.dot(
            rotation_matrix, np.vstack([x_centered, y_centered])
        )

        # Возвращаем график в исходное положение
        rotated_points = xy_rotated[0] + np.mean(points)
        rotated_array = xy_rotated[1] + np.mean(data_array)

        return rotated_points, rotated_array


---

### Filters Class

Объект класса `Filters` содержит методы для фильтрации словарей массивов данных.

При инициализации объекта класса `Filters` создается словарь методов фильтрации, где ключами являются имена методов,
а значениями - сами методы.

    class Filters:
    
        def __init__(self):
            self.methods = {
                "std": self.remove_by_std,
                "forbidden_columns": self.remove_by_forbidden_columns
            }
    
#### Метод `filter_by`
Метод `filter_by` вызывает соответствующий метод фильтрации из словаря `methods` и передает ему аргументы.

        def filter_by(self, method: str, *args, **kwargs):
            return self.methods[method](*args, **kwargs)
    
#### Метод `remove_by_std`
Метод `remove_by_std` проверяет имеются ли в массиве данных `Array` нулевые стандартные отклонения. Возвращает `True`,
если стандартное отклонение равно нулю, иначе `False`.

        @staticmethod
        def remove_by_std(data_array: Array, *args, **kwargs):
            if data_array.is_numeric:
                return True if data_array.values.std() == 0 else False
            else:
                return False
    
#### Метод `remove_by_forbidden_columns`
Метод `remove_by_forbidden_columns` проверяет присутствует ли идентификатор массива данных (транскрипция) в списке запрещенных
столбцов `forbidden_columns`, которые определены в словаре массивов данных `SourceDataset`.
Возвращает `True`, если транскрипция присутствует в списке запрещенных столбцов, иначе `False`.

        @staticmethod
        def remove_by_forbidden_columns(
            data_array: Array,
            source_data: SourceDataset,
            *args,
            **kwargs,
        ):
            if (data_array.transcription.name
                    in source_data.settings.forbidden_columns):
                return True
            else:
                return False

---

# 7. Фабрика сборки кастомных пайплайнов. Обработчики данных.

Модуль `handlers` содержит логику по обработке данных, например, применение различных алгоритмов анализа,
обработки, и т.д. Фабрика `HandlersFactory` используется для создания экземпляров этих обработчиков (handlers) с определенными
настройками и параметрами.

Инициализация экземпляра `HandlersFactory` из ранее созданных компонент запуска процесса обработки файлов:

    handlers_factory = HandlersFactory(
        features_generator=features_generator,
        filters=filters,
        secondary_functions=secondary_functions,
        segments_aggregator=segments_aggregator,
        materials=materials
    )

---

### HandlersFactory Class

Класс `HandlersFactory` является фабрикой для создания обработчиков данных, которые называются "handlers".

Каждый обработчик возвращает словарь агрегированных сегментов по данным.
быть обработка признаков, применение фильтров, агрегация сегментов и т. д.

В конструкторе `__init__` фабрика получает экземпляры компонентов запуска процесса обработки файлов:
`features_generator`, `filters`, `secondary_functions` и `segments_aggregator`. Затем она создает словарь, где ключами
являются наименования обработчиков для вызова, а значениями - соответствующие им обработчики.

    class HandlersFactory:
    
        def __init__(
            self,
            features_generator: FeaturesGenerator,
            filters: Filters,
            materials: Materials,
            secondary_functions: SecondaryFunctions,
            segments_aggregator: SegmentsAggregator
        ):
            handler_setup = {
                "features_generator": features_generator,
                "filters": filters,
                "secondary_functions": secondary_functions,
                "segments_aggregator": segments_aggregator,
                "materials": materials
            }
            self.handlers_dict = {
                BASEHandler.source: BASEHandler(**handler_setup),
                BASEWITHPOINTSHandler.source: BASEHandler(**handler_setup),
                WBFPIROHandler.source: WBFPIROHandler(**handler_setup),
                WBFSINGLEHandler.source: WBFSINGLEHandler(**handler_setup)
            }
    
#### Метод `get_handler`
Метод `get_handler` принимает ключ обработчика и возвращает соответствующий обработчик данных из словаря `handlers_dict`.

        def get_handler(self, source: str) -> SourceHandler:
            return self.handlers_dict[source]


### SourceHandler Class

Класс `SourceHandler` является абстрактным базовым классом для всех обработчиков данных.
Он определяет общий интерфейс для всех обработчиков, которые должны быть реализованы в подклассах.

    from abc import ABC, abstractmethod

    class SourceHandler(ABC):
        source: str # источник данных 
    
#### Метод `__init__`
Метод `__init__`: инициализирует обработчик данных. Принимает все компоненты запуска процесса обработки файлов:
`features_generator`, `filters`, `secondary_functions` и `segments_aggregator`, которые будут использоваться в процессе обработки данных. 

        @abstractmethod
        def __init__(
            self,
            features_generator: FeaturesGenerator,
            filters: Filters,
            secondary_functions: SecondaryFunctions,
            segments_aggregator: SegmentsAggregator
        ):
            self.segments_aggregator = segments_aggregator
            self.secondary_functions = secondary_functions
            self.feature_generator = features_generator
            self.filters = filters
    
#### Метод `process_pipeline`
Метод `process_pipeline` является абстрактным методом, который должен быть реализован в подклассах. Принимает:
- идентификатор заготовки (billet_id);
- данные по заготовке в виде DataFrame (data);
- настройки источника данных (settings).

Этот метод выполняет обработку данных для конкретного источника конкретной заготовки с использованием компонентов обработки, переданных в конструкторе.

        @abstractmethod
        def process_pipeline(
            self, billet_id: str, data: pd.DataFrame, settings: SourceSettings
        ):
            ...


### BASEHandler Class

Получение объекта `BASEHandler` через атрибут `source` = `BASE`:
  
    handler = handlers_factory.get_handler("BASE")

`BASEHandler` является базовым обработчиком данных и реализует общий шаблон обработки для всех типов данных. Этот
класс может быть настроен и расширен для обработки конкретных типов данных или источников.
`BASEHandler` является подклассом `SourceHandler`.

    class BASEHandler(SourceHandler):
        source = "BASE"
    
        def __init__(
            self,
            features_generator: FeaturesGenerator,
            filters: Filters,
            materials: Materials,
            secondary_functions: SecondaryFunctions,
            segments_aggregator: SegmentsAggregator,
        ):
            self.materials = materials
            self.segments_aggregator = segments_aggregator
            self.secondary_functions = secondary_functions
            self.feature_generator = features_generator
            self.filters = filters
    
#### Метод `process_pipeline`

        def process_pipeline(
            self, billet_id: str, data: pd.DataFrame, settings: SourceSettings
        ) -> AggregatedSourceDict:
            source_data = SourceDataset(settings)
    
Подготовительный пайплайн: В этом этапе выполняются несколько предварительных операций над исходными данными, такие как:

- Преобразование дата-фрейма к словарю массивов данных;
- Сортировка данных по возрастанию значений столбца `billet points` (координат заготовки);
- Апроксимация значений столбца `billet points` к заданной размерности

            # Подготовительный пайплайн
            source_data.append_dataframe_to_source_data(data)
            source_data = (
                self.secondary_functions.
                sort_dataset_ascending_by_billet(source_data)
            )
            source_data = self.secondary_functions.approximate_billet_by_bounds(
                source_data
            )

Присвоение к транскрипции всех массивов данных наименования рабочего центра (`mill` (стан), `ccm` (МНЛЗ)) по источнику данных
(агрегату), которые указаны во вспомогательных компонентах `materials = {'workcenters': 'agregator/run/materials/workcenters.json'}`:

            with open(self.materials.PATHS['workcenters']) as handle:
                workcenters = json.load(handle)
            source_data = (
                self.secondary_functions.append_workcenter_to_transcription(
                    source_data, workcenters
                )
            )

Определение базового пайплайна:

            # Базовый пайплайн
            source_data = self.filter_data(source_data)
            source_data = self.generate_features(source_data)
            aggregated_source = self.calculate_aggregations(source_data)
            return aggregated_source
    
#### Метод `filter_data`
Метод `filter_data` удаляет массивы данных, имена которых попали в список запрещенных столбцов `forbidden_columns`.

        def filter_data(self, source_data: SourceDataset) -> SourceDataset:
            """Фильтрация данных"""
            filtered_columns = []
            for data_array in source_data:
                if data_array.is_billet:
                    continue
                for method in [*source_data.settings.filtration_methods,
                               "forbidden_columns"]:
                    is_bad = self.filters.filter_by(
                        method=method,
                        data_array=data_array,
                        source_data=source_data
                    )
                    if is_bad:
                        filtered_columns.append(data_array.transcription)
            source_data.remove_arrays(filtered_columns)
            return source_data
    
#### Метод `generate_features`
Метод `generate_features` создает вторичные признаки на основе заданных методов [abs], которые указываются
в конфигурационном файле `settings`. Каждый новый признак добавляется в словарь массивов данных `source_data`.

        def generate_features(self, source_data: SourceDataset) -> SourceDataset:
            """Генерация вторичных фичей"""
            new_arrays = []
            for method in source_data.settings.secondary_features:
                for data_array in source_data:
                    if data_array.is_billet or not data_array.is_numeric:
                        continue
                    new_arrays.append(
                        self.feature_generator.get_method_values(
                            method, data_array
                        )
                    )
            for new_array in new_arrays:
                source_data.append_array_to_source_data(new_array)
            return source_data
    
#### Метод `calculate_aggregations`
Метод `calculate_aggregations` агрегирует данные по сегментам. Для каждого сегмента он создает объект `AggregatedSegment`,
содержащий агрегированные значения. Для каждого массива данных метод агрегации применяется к соответствующему сегменту.
Полученные значения добавляются в объект `AggregatedSegment`. В конце метод возвращает объект `AggregatedSourceDict`, содержащий
агрегированные сегменты.


        def calculate_aggregations(
            self, source_data: SourceDataset
        ) -> AggregatedSourceDict:
            """Аггрегация данных по сегментам"""

Создание объекта `AggregatedSourceDict`, который будет содержать агрегированные данные по сегментам.

            aggregated_source = AggregatedSourceDict(source_data.settings)

Итерация по сегментам массивов данных, например, по следущим сегментам:

    [1, 0, 4000, ['LNK100_1']]
    [2, 4000, 22000, ['LNK100_1']]
    [3, 22000, 32000, ['LNK100_1']]
    [4, 32000, 42000, ['LNK100_1']]
    [5, 42000, "end", ['LNK100_1']]

Для каждого сегмента создается новый объект `AggregatedSegment` и производится агрегирование массивов данных по данному сегменту:

            for segment_id, segment in source_data.settings.segments.items():
                aggregated_segment = AggregatedSegment(
                    segment_id, segment.start_point, segment.end_point
                )

Итерация по массивам данных, за исключением данных, связанных с заготовкой, по текущему сегменту:

                for data_array in source_data:
                    if data_array.is_billet:
                        continue

Для каждого массива данных мы рассчитываем агрегированные значения по указанным методам агрегации в `source_data` в рамках
текущего сегмента:

                    for method in source_data.settings.aggregation_methods:

Создаем транскрипцию, добавляя информацию о сегменте и методе агрегации:

                        transcription = data_array.transcription.add_tags(
                            {
                                "sector_range": [
                                    f"{segment.start_point}_{segment.end_point}"
                                ],
                                "aggregation": [f"{method}"]
                            }
                        )

Получаем значения, попавшие в текущий сегмент:

                        segment_values = (
                            self.segments_aggregator.return_segment_values(
                                segment, source_data.billet_array(), data_array
                            )
                        )
                        segment_values = segment_values[~pd.isnull(segment_values)]

Если типы значений массива данных не числовые, то данному сегменту присваивается флаг `is_bad`, с подписью "Not numeric":

                        if not data_array.is_numeric:
                            aggregated_value = AggregatedValue(
                                segment_id,
                                transcription,
                                None,
                                True,
                                "Not numeric",
                            )

Если для данного сегмента не было найдено значений в массиве данных, то данному сегменту присваивается флаг `is_bad`, с подписью "Empty":

                        elif len(segment_values) == 0:
                            aggregated_value = AggregatedValue(
                                segment_id, transcription, None, True, "Empty"
                            )

В ином случае сегменту присваивается посчитанное агрегированное значение с прочими данными segment_id, transcription, method:

                        else:
                            aggregated_value = (
                                self.segments_aggregator.get_method_value(
                                    segment_id, transcription, method,
                                    segment_values
                                )
                            )
                        aggregated_segment.append_value([aggregated_value])

Все агрегированные сегменты добавляются в словарь сегментов `aggregated_source`.

                aggregated_source.append_segment(aggregated_segment)
            return aggregated_source

Пример: для следующих конфигурационных параметров:

- Сегменты:[1, 0, 3, []]
- Параметры источника:

| Vert1000 | Vert1500 | Vert2000 | Vert3000 | Hor1000 | Hor1500 | Hor2000 | Torsion |
|----------|----------|----------|----------|---------|---------|---------|---------|

- Методы фильтрации: ["std"]
- Вспомогательные функции: ["abs"]
- Методы агрегации: ['max', 'min', 'median']
- Интерполяция: 103


Пример построения пайплайна для проверки работы обработчика `BASEHandler`:

    # Компоненты запуска процесса обработки файлов
    PATH_TO_MATERIALS=r"exploration\agregator\run\materials\*"
    
    materials = Materials(
        {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in glob(PATH_TO_MATERIALS)
        } if PATH_TO_MATERIALS else None
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
    
    # ID заготовки
    billet_id = "Л210421011"
    
    file_path = "testing/20240308000407_21046X404_IPSRNK_1_L.csv"
    df = pd.read_csv(file_path, sep=';')
    for col in df.columns:
        if col == "BilletPoints":
            df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce").fillna(0).astype(float)
        else:
            df[col] = pd.to_numeric(df[col].replace(",", "."), errors="coerce").fillna(0).astype(float)
    
    data = df  # Ваш DataFrame
    
    # Создание объекта SourceSettings
    settings = SourceSettings(
        source="LNK100",
        type="target",
        handler="BASE",
        rolling_number="1",
        encoding="UTF-8",
        is_single=False,
        main_folder="\\ZSMK-9684-001\Data\2023",
        key_folder="ipsrnk",
        nested_folders="rail_points_files",
        filename_key="",
        interpolation_type="by_value",
        interpolation=103,
        segments={
                "LNK100_1": Segment(start_point=0, end_point=3, target_segments=[])
            },
        forbidden_columns=[],
        filtration_methods=["std"],
        secondary_features=["abs"],
        aggregation_methods=['max', 'min', 'median'],
        billet_column="BilletPoints",
        convert_columns={}
    )
    
    # Получение объекта BASEHandler
    handler = handlers_factory.get_handler("BASE")
    
    # Запуск обработки данных с помощью BASEHandler
    seg = handler.process_pipeline(billet_id, data, settings)

Обработчик `BASEHandler` вернет следующий словарь агрегированных сегментов:

    {'mill_LNK100_1_Hor1500_[i_0.0_103.0]_0_3_max': mill_LNK100_1_Hor1500_[i_0.0_103.0]_0_3_max, value=443.0,
     'mill_LNK100_1_Hor1500_[i_0.0_103.0]_0_3_min': mill_LNK100_1_Hor1500_[i_0.0_103.0]_0_3_min, value=223.0,
     'mill_LNK100_1_Hor1500_[i_0.0_103.0]_0_3_median': mill_LNK100_1_Hor1500_[i_0.0_103.0]_0_3_median, value=283.0,
     'mill_LNK100_1_Hor2000_[i_0.0_103.0]_0_3_max': mill_LNK100_1_Hor2000_[i_0.0_103.0]_0_3_max, value=719.0,
     'mill_LNK100_1_Hor2000_[i_0.0_103.0]_0_3_min': mill_LNK100_1_Hor2000_[i_0.0_103.0]_0_3_min, value=340.0,
     'mill_LNK100_1_Hor2000_[i_0.0_103.0]_0_3_median': mill_LNK100_1_Hor2000_[i_0.0_103.0]_0_3_median, value=492.0,
      ...
     'mill_LNK100_1_Vert3000_[i_0.0_103.0]_0_3_max': mill_LNK100_1_Vert3000_[i_0.0_103.0]_0_3_max, value=397.0,
     'mill_LNK100_1_Vert3000_[i_0.0_103.0]_0_3_min': mill_LNK100_1_Vert3000_[i_0.0_103.0]_0_3_min, value=240.0,
     'mill_LNK100_1_Vert3000_[i_0.0_103.0]_0_3_median': mill_LNK100_1_Vert3000_[i_0.0_103.0]_0_3_median, value=327.0,
     'mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_max': mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_max, value=238.0,
     'mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_min': mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_min, value=122.0,
     'mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_median': mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_median, value=150.0}


### BASEWITHPOINTSHandler Class

Получение объекта `BASEWITHPOINTSHandler` через атрибут `source` = `BASE`:
  
    handler = handlers_factory.get_handler("BASE")

Обработчик `BASEWITHPOINTSHandler` похож на `BASEHandler`, но с некоторыми изменениями в методе `generate_features`
и `calculate_aggregations`. Ниже разберем именно эти методы.

#### Метод `generate_features`
Метод `generate_features` также генерирует вторичные признаки, но с небольшим изменением. Он принимает два дополнительных аргумента MIN и MAX,
что позволяет использовать метод [norm] для нормализации данных.

     def generate_features(
        self, source_data: SourceDataset, MIN, MAX
    ) -> SourceDataset:
        """Генерация вторичных фичей"""
        new_arrays = []
        for method in source_data.settings.secondary_features:
            for data_array in source_data:
                if data_array.is_billet or not data_array.is_numeric:
                    continue
                new_arrays.append(
                    self.feature_generator.get_method_values(
                        method, data_array, MIN, MAX
                    )
                )
        for new_array in new_arrays:
            source_data.append_array_to_source_data(new_array)
        return source_data

#### Метод `calculate_aggregations`
Метод `calculate_aggregations` также очень похож на аналогичный метод в `BASEHandler`. Он проходит по каждому сегменту и для
каждого массива данных агрегирует значения в соответствии с методами агрегации, указанными в настройках.
Однако здесь есть небольшое изменение: помимо значений массива данных, которые попали в сегмент, он также принимает координаты 
точек (points) `billet points` заготовки в интервале от 0 до 3, что позволяет учитывать расположение значений внутри сегмента при агрегации,
а также расчитать тангенс угла наклона прямой [tg], аппроксимирующей зависимость между данными массива и координатами заготовки.

    ...
    points, segment_values = (
        self.segments_aggregator.return_segment_values(
            segment, source_data.billet_array(), data_array
        )
    )
    nan_indexes = np.where(np.isnan(segment_values))
    segment_values = segment_values[~np.isnan(segment_values)]
    points = np.delete(points, nan_indexes)
    segment_values = segment_values[~pd.isnull(segment_values)]
    ...

### WBFPIROHandler Class

Получение объекта `WBFPIROHandler` через атрибут `source` = `WBFPIRO`:
  
    handler = handlers_factory.get_handler("WBFPIRO")

Обработчик `WBFPIROHandler` похож на `BASEHandler`, но с некоторыми изменениями в методе `process_pipeline`
и `calculate_aggregations`. Ниже разберем именно эти методы.

#### Метод `process_pipeline`
Здесь важно отметить следующие изменения:
- Производится конвертация столбцов с датой в числовой формат с помощью метода `convert_date_columns_to_numeric`.

        ...
        source_data = self.secondary_functions.convert_date_columns_to_numeric(
            source_data, [{
                "name": "moment"
            }]
        )
        ...

- Формируются дополнительные сегменты с помощью метода `cut_wbf_pirometer_signals` на основе сигнала согласно заданным окнам, указанным в
`wbf_piro_cutter_settings`:

        ...
        with open(self.materials.PATHS['wbf_piro_cutter_settings']) as handle:
            wbf_piro_cutter_settings = json.load(handle)
        source_data = self.secondary_functions.cut_wbf_pirometer_signals(
            source_data, wbf_piro_cutter_settings
        )
        ...

#### Метод `calculate_aggregations`
Метод `calculate_aggregations` в `WBFPIROHandler` и `BASEHandler` в целом идентичен, за исключением формирования тега `sector_range` в транскрипции.

    ...
    transcription = data_array.transcription.add_tags(
        {
            "sector_range": [f"{segment_id}"],
            "aggregation": [f"{method}"]
        }
    )
    ...

Пример построения пайплайна для проверки работы обработчика `WBFPIROHandler`:

    # Компоненты запуска процесса обработки файлов
    PATH_TO_MATERIALS=r"exploration\agregator\run\materials\*"
    
    materials = Materials(
        {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in glob(PATH_TO_MATERIALS)
        } if PATH_TO_MATERIALS else None
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
    
    # Параметры для обработки
    billet_id = "Л210421011"
    
    file_path = 'testing/20240307170148_XX-Л210444041_WBF_1_T2.csv'
    df = pd.read_csv(file_path, sep=';')
    for col in df.drop("moment", axis=1).columns:
        df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")
    df["moment"] = pd.to_datetime(df["moment"])
    
    data = df  # Ваш DataFrame
    
    # Создание объекта SourceSettings
    settings = SourceSettings(
        source="WBF_PIRO",
        type="feature",
        handler="WBFPIRO",
        rolling_number="1",
        encoding="UTF-8",
        is_single=False,
        main_folder="\\ZSMK-9684-001\Data\2023",
        key_folder="wbf",
        nested_folders="billet_pirometer_files",
        filename_key="",
        interpolation_type="",
        interpolation="",
        segments={
                "WBF_PIRO_1": Segment(start_point=0, end_point="end", target_segments=["LNK100_1"])
            },
        forbidden_columns=["BEAM_VERT_ENC", "BEAM_HOR_ENC_POS", "DM_ENC_HOR_POS", "DML_VERT_POS", "DMR_VERT_POS"],
        filtration_methods=[],
        secondary_features=[],
        aggregation_methods=['max', 'min','median','mean'],
        billet_column="moment",
        convert_columns={}
    )
    
    # Путь к файлу JSON до параметров окон для формирования дополнительных сегментов
    file_path = "exploration/agregator/run/materials/wbf_piro_cutter_settings.json"
    with open(file_path, "r") as file:
        piro_settings = json.load(file)
    
    # Получение объекта WBFPIROHandler
    handler = handlers_factory.get_handler("WBFPIRO")
    
    # Запуск обработки данных с помощью WBFPIROHandler
    seg = handler.process_pipeline(billet_id, data, settings)

Обработчик `WBFPIROHandler` вернет следующий словарь агрегированных сегментов:

    {'mill_WBF_PIRO_1_TEMP_PIR3_ni_WBF_PIRO_1_max': mill_WBF_PIRO_1_TEMP_PIR3_ni_WBF_PIRO_1_max, value=1187.239501953125,
     'mill_WBF_PIRO_1_TEMP_PIR3_ni_WBF_PIRO_1_min': mill_WBF_PIRO_1_TEMP_PIR3_ni_WBF_PIRO_1_min, value=1107.1539306640625,
     'mill_WBF_PIRO_1_TEMP_PIR3_ni_WBF_PIRO_1_median': mill_WBF_PIRO_1_TEMP_PIR3_ni_WBF_PIRO_1_median, value=1183.767333984375,
     'mill_WBF_PIRO_1_TEMP_PIR3_ni_WBF_PIRO_1_mean': mill_WBF_PIRO_1_TEMP_PIR3_ni_WBF_PIRO_1_mean, value=1165.8901766886754,
     'mill_WBF_PIRO_1_TEMP_PIR1_ni_WBF_PIRO_1_max': mill_WBF_PIRO_1_TEMP_PIR1_ni_WBF_PIRO_1_max, value=1187.5867919921875,
     'mill_WBF_PIRO_1_TEMP_PIR1_ni_WBF_PIRO_1_min': mill_WBF_PIRO_1_TEMP_PIR1_ni_WBF_PIRO_1_min, value=1083.94091796875,
     'mill_WBF_PIRO_1_TEMP_PIR1_ni_WBF_PIRO_1_median': mill_WBF_PIRO_1_TEMP_PIR1_ni_WBF_PIRO_1_median, value=1185.9375,
     'mill_WBF_PIRO_1_TEMP_PIR1_ni_WBF_PIRO_1_mean': mill_WBF_PIRO_1_TEMP_PIR1_ni_WBF_PIRO_1_mean, value=1161.4180109833446,
     'mill_WBF_PIRO_1_TEMP_PIR2_ni_WBF_PIRO_1_max': mill_WBF_PIRO_1_TEMP_PIR2_ni_WBF_PIRO_1_max, value=1148.495361328125,
     'mill_WBF_PIRO_1_TEMP_PIR2_ni_WBF_PIRO_1_min': mill_WBF_PIRO_1_TEMP_PIR2_ni_WBF_PIRO_1_min, value=1090.4803466796875,
     'mill_WBF_PIRO_1_TEMP_PIR2_ni_WBF_PIRO_1_median': mill_WBF_PIRO_1_TEMP_PIR2_ni_WBF_PIRO_1_median, value=1140.3935546875,
     'mill_WBF_PIRO_1_TEMP_PIR2_ni_WBF_PIRO_1_mean': mill_WBF_PIRO_1_TEMP_PIR2_ni_WBF_PIRO_1_mean, value=1135.0946023830923}


### WBFSINGLEHandler Class

Получение объекта `WBFSINGLEHandler` через атрибут `source` = `WBFSINGLE`:
  
    handler = handlers_factory.get_handler("WBFSINGLE")

Обработчик `WBFSINGLEHandler` работает с датафреймом, где все данные по заготовкам собраны в одном CSV файле и каждая строка в этом файле представляет отдельную заготовку:

| PIECE_PK      | FURNACE_FK | billet_number | CHARGING_TIME       | TOT_X_NODE | TOT_Y_NODE | TOT_Z_NODE | X_POS | X_COORD | IS_DISCHARGED | ... | Z1_LV2_IN_percent | Z2_LV2_IN_percent | Z3_LV2_IN_percent | Z4_LV2_IN_percent | Z5_LV2_IN_percent | Z6_LV2_IN_percent | Z7_LV2_IN_percent | Z8_LV2_IN_percent | Z9_LV2_IN_percent | Z10_LV2_IN_percent |
|---------------|------------|---------------|---------------------|------------|------------|------------|-------|---------|---------------|-----|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|--------------------|
| 2.403071e+19  | 1          | Л210444041    | 2024-03-07 19:36:25 | 100        | 7          | 7          | 2714.0| 6678.0  | Y             | ... | 100.0             | 100.0             | 100.0             | 100.0             | 100.0             | 100.0             | 100.000000        | 100.000000        | 100.000000        | 100.000000         |
| 2.403071e+19  | 1          | Л210421011    | 2024-03-07 19:38:53 | 100        | 7          | 7          | 599.0 | 4540.5  | Y             | ... | 100.0             | 100.0             | 100.0             | 100.0             | 100.0             | 100.0             | 100.000000        | 100.000000        | 100.000000        | 100.000000         |
| 2.403071e+19  | 1          | Л210422011    | 2024-03-07 19:41:17 | 100        | 7          | 7          | 2717.0| 6666.5  | Y             | ... | 100.0             | 100.0             | 100.0             | 100.0             | 100.0             | 100.0             | 100.000000        | 100.000000        | 100.000000        | 100.000000         |


Обработчик `WBFSINGLEHandler` полностью повторяет логику работы метода `filter_data` обработчика `BASEHandler`, также не имеет
метода `generate_features`. Ниже разберем реализацию методов `process_pipeline` и `calculate_aggregations` данного обработчика.

#### Метод `process_pipeline`
Метод `process_pipeline` принимает исходные данные, обрабатывает их и возвращает словарь агрегированных сегментов.

    def process_pipeline(
        self, billet_id: str, data: pd.DataFrame, settings: SourceSettings
    ) -> AggregatedSourceDict:

        # Фильтрация данных по номеру заготовки (billet_id)

        data = data[data[settings.billet_column
                         ].apply(lambda x: x[:-1] in billet_id)]

        # Сортировка отфильтрованных данных по времени `CHARGING_TIME` в порядке возрастания

        data = data.sort_values(by=["CHARGING_TIME"])

        # Создание объекта `SourceDataset` с использованием переданных настроек.

        source_data = SourceDataset(settings)

        # Подготовительный пайплайн
        source_data.append_dataframe_to_source_data(data)

        # Базовый пайплайн
        source_data = self.filter_data(source_data)
        aggregated_source = self.calculate_aggregations(source_data)
        return aggregated_source

#### Метод `calculate_aggregations`
Метод `calculate_aggregations` в обработчике `WBFSINGLEHandler` не расчитывает агрегацию по сегменту, а агрегированному значению
сегмента присваивается единственное значение массива, в идентификаторе сегмента тегу `sector_range` присваивается `single`.

    def calculate_aggregations(
        self, source_data: SourceDataset
    ) -> AggregatedSourceDict:
        """Аггрегация данных по сегментам"""
        aggregated_source = AggregatedSourceDict(source_data.settings)
        for segment_id, segment in source_data.settings.segments.items():
            aggregated_segment = AggregatedSegment(
                segment_id, segment.start_point, segment.end_point
            )
            for data_array in source_data:
                if data_array.is_billet:
                    continue
                transcription = data_array.transcription.add_tags(
                    {"sector_range": ["single"]}
                )
                if not data_array.is_numeric:
                    aggregated_value = AggregatedValue(
                        segment_id, transcription, None, True, "Not numeric"
                    )
                else:
                    aggregated_value = AggregatedValue(
                        segment_id, transcription, float(data_array.values[0])
                    )
                aggregated_segment.append_value([aggregated_value])
            aggregated_source.append_segment(aggregated_segment)
        return aggregated_source


Пример построения пайплайна для проверки работы обработчика `WBFSINGLEHandler`:

    # Компоненты запуска процесса обработки файлов
    PATH_TO_MATERIALS=r"exploration\agregator\run\materials\*"
    
    materials = Materials(
        {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in glob(PATH_TO_MATERIALS)
        } if PATH_TO_MATERIALS else None
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
    
    # Параметры для обработки
    billet_id = "Л210421011"
    
    file_path = 'testing/20240308000000_20240309000000_WBF.csv'
    # Чтение файла с указанием кодировки ANSI
    df = pd.read_csv(file_path, sep=';', encoding='ANSI')
    forbidden_cols = ["billet_number", "CHARGING_TIME", "IS_DISCHARGED"]
    for col in df.columns:
        if col not in forbidden_cols:
            # Если тип столбца - строка, заменяем запятые на точки и преобразуем в числовой тип
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    data = df  # Ваш DataFrame
    
    # Создание объекта SourceSettings
    settings = SourceSettings(
        source="WBF_sgl",
        type="feature",
        handler="WBFSINGLE",
        rolling_number="1",
        encoding="ANSI",
        is_single=True,
        main_folder="\\ZSMK-9684-001\Data\2023",
        key_folder="wbf",
        nested_folders="",
        filename_key="",
        interpolation_type="",
        interpolation="",
        segments={
                "WBF_sgl_1": Segment(start_point=0, end_point=0, target_segments=["LNK100_1"])
            },
        forbidden_columns=[],
        filtration_methods=[],
        secondary_features=[],
        aggregation_methods=[],
        billet_column="billet_number",
        convert_columns={}
    )
    
    # Получение объекта WBFSINGLEHandler
    handler = handlers_factory.get_handler("WBFSINGLE")
    
    # Запуск обработки данных с помощью WBFSINGLEHandler
    seg = handler.process_pipeline(billet_id, data, settings)


Обработчик `WBFSINGLEHandler` вернет следующий словарь агрегированных сегментов:

    {'WBF_sgl_1_PIECE_PK_ni_single': WBF_sgl_1_PIECE_PK_ni_single, value=2.4030712380000006e+19,
     'WBF_sgl_1_FURNACE_FK_ni_single': WBF_sgl_1_FURNACE_FK_ni_single, value=1.0,
     'WBF_sgl_1_CHARGING_TIME_ni_single': WBF_sgl_1_CHARGING_TIME_ni_single, value=None,
     'WBF_sgl_1_TOT_X_NODE_ni_single': WBF_sgl_1_TOT_X_NODE_ni_single, value=100.0,
     'WBF_sgl_1_TOT_Y_NODE_ni_single': WBF_sgl_1_TOT_Y_NODE_ni_single, value=7.0,
     'WBF_sgl_1_TOT_Z_NODE_ni_single': WBF_sgl_1_TOT_Z_NODE_ni_single, value=7.0,
      ...
     'WBF_sgl_1_Z6_LV2_IN_percent_ni_single': WBF_sgl_1_Z6_LV2_IN_percent_ni_single, value=100.0,
     'WBF_sgl_1_Z7_LV2_IN_percent_ni_single': WBF_sgl_1_Z7_LV2_IN_percent_ni_single, value=100.0,
     'WBF_sgl_1_Z8_LV2_IN_percent_ni_single': WBF_sgl_1_Z8_LV2_IN_percent_ni_single, value=100.0,
     'WBF_sgl_1_Z9_LV2_IN_percent_ni_single': WBF_sgl_1_Z9_LV2_IN_percent_ni_single, value=100.0,
     'WBF_sgl_1_Z10_LV2_IN_percent_ni_single': WBF_sgl_1_Z10_LV2_IN_percent_ni_single, value=100.0}

---

# 8. Класс для сопоставления данных фичей с таргетом

Класс `Matcher` используется для сопоставления данных фичей с таргетом. После агрегации данных и получения фичей,
необходимо связать эти данные с соответствующими целевыми значениями, чтобы подготовить их для дальнейшего обучения модели.

        # Класс для сопоставления данных фичей с таргетом
        matcher = Matcher(sources_settings)

### Matcher Class

Класс `Matcher` инициализируется с помощью словаря `sources_settings`, например:

    LNK100_settings = SourceSettings(
        source="LNK100",
        type="target",
        handler="BASE",
        rolling_number="1",
        encoding="UTF-8",
        is_single=False,
        main_folder="\\ZSMK-9684-001\Data\2023",
        key_folder="ipsrnk",
        nested_folders="rail_points_files",
        filename_key="",
        interpolation_type="by_value",
        interpolation=103,
        segments={
                "LNK100_1": Segment(start_point=0, end_point=3, target_segments=[])
            },
        forbidden_columns=[],
        filtration_methods=["std"],
        secondary_features=["abs"],
        aggregation_methods=['max', 'min', 'median'],
        billet_column="BilletPoints",
        convert_columns={}
    )
    
    WBF_PIRO_settings = SourceSettings(
        source="WBF_PIRO",
        type="feature",
        handler="WBFPIRO",
        rolling_number="1",
        encoding="UTF-8",
        is_single=False,
        main_folder="\\ZSMK-9684-001\Data\2023",
        key_folder="wbf",
        nested_folders="billet_pirometer_files",
        filename_key="",
        interpolation_type="",
        interpolation="",
        segments={
                "WBF_PIRO_1": Segment(start_point=0, end_point="end", target_segments=["LNK100_1"])
            },
        forbidden_columns=["BEAM_VERT_ENC", "BEAM_HOR_ENC_POS", "DM_ENC_HOR_POS", "DML_VERT_POS", "DMR_VERT_POS"],
        filtration_methods=[],
        secondary_features=[],
        aggregation_methods=['max', 'min','median','mean'],
        billet_column="moment",
        convert_columns={}
    )
    
    WBF_sgl_settings = SourceSettings(
        source="WBF_sgl",
        type="feature",
        handler="WBFSINGLE",
        rolling_number="1",
        encoding="ANSI",
        is_single=True,
        main_folder="\\ZSMK-9684-001\Data\2023",
        key_folder="wbf",
        nested_folders="",
        filename_key="",
        interpolation_type="",
        interpolation="",
        segments={
                "WBF_sgl_1": Segment(start_point=0, end_point=0, target_segments=["LNK100_1"])
            },
        forbidden_columns=[],
        filtration_methods=[],
        secondary_features=[],
        aggregation_methods=[],
        billet_column="billet_number",
        convert_columns={}
    )
    
    sources_settings = {
        "LNK100": LNK100_settings,
        "WBF_PIRO": WBF_PIRO_settings,
        "WBF_sgl": WBF_sgl_settings
    }

где ключами являются имена источников данных, а значениями - объекты `SourceSettings`, содержащие настройки для каждого источника данных.

      class Matcher:
      
          def __init__(self, sources_settings: Dict[str, SourceSettings]):
              self.sources_settings = sources_settings

В переменной `target_name` сохраняется имя целевого источника данных. Это имя извлекается из словаря `sources_settings` путем итерации по его элементам и выбора первого источника данных с типом `target`.

              self.target_name = [
                  name for name, src in self.sources_settings.items()
                  if src.type == "target"
              ][0]
      
#### Метод `match_features_to_target` 
Метод `match_features_to_target` принимает на вход словарь словарей агрегированных сегментов по источникам данных, например 

    all_aggregated_sources = {
        "LNK100": seg_LNK100,
        "WBF_PIRO": seg_WBF_PIRO,
        "WBF_sgl": seg_WBF_sgl
    }

где

      all_aggregated_sources["LNK100"]["LNK100_1"] =>
      
    {'mill_LNK100_1_Hor1500_[i_0.0_103.0]_0_3_max': mill_LNK100_1_Hor1500_[i_0.0_103.0]_0_3_max, value=443.0,
     'mill_LNK100_1_Hor1500_[i_0.0_103.0]_0_3_min': mill_LNK100_1_Hor1500_[i_0.0_103.0]_0_3_min, value=223.0,
     'mill_LNK100_1_Hor1500_[i_0.0_103.0]_0_3_median': mill_LNK100_1_Hor1500_[i_0.0_103.0]_0_3_median, value=283.0,
     'mill_LNK100_1_Hor2000_[i_0.0_103.0]_0_3_max': mill_LNK100_1_Hor2000_[i_0.0_103.0]_0_3_max, value=719.0,
     'mill_LNK100_1_Hor2000_[i_0.0_103.0]_0_3_min': mill_LNK100_1_Hor2000_[i_0.0_103.0]_0_3_min, value=340.0,
     'mill_LNK100_1_Hor2000_[i_0.0_103.0]_0_3_median': mill_LNK100_1_Hor2000_[i_0.0_103.0]_0_3_median, value=492.0,
      ...
     'mill_LNK100_1_Vert3000_[i_0.0_103.0]_0_3_max': mill_LNK100_1_Vert3000_[i_0.0_103.0]_0_3_max, value=397.0,
     'mill_LNK100_1_Vert3000_[i_0.0_103.0]_0_3_min': mill_LNK100_1_Vert3000_[i_0.0_103.0]_0_3_min, value=240.0,
     'mill_LNK100_1_Vert3000_[i_0.0_103.0]_0_3_median': mill_LNK100_1_Vert3000_[i_0.0_103.0]_0_3_median, value=327.0,
     'mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_max': mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_max, value=238.0,
     'mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_min': mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_min, value=122.0,
     'mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_median': mill_LNK100_1_Hor1000_[i_0.0_103.0]_0_3_median, value=150.0}

И выполняет сопоставление данных сегментов фичей с целевым источником данных.

          def match_features_to_target(
              self, all_aggregated_sources: Dict[str, AggregatedSourceDict]
          ) -> (Dict[str, AggregatedSegment], Dict[str, Dict[str, float]]):
              targets_segments = {
                  seg_key: all_aggregated_sources[self.target_name][seg_key]
                  for seg_key in self.sources_settings[self.target_name
                                                       ].segments.keys()
              }
              for source_name, source_values in all_aggregated_sources.items():
                  if source_values.is_target():
                      continue
                  for segment_name, segment_values in source_values.items():
                      segment_settings = source_values.settings(
                      ).segments[segment_name]
                      for target_name in segment_settings.target_segments:
                          targets_segments[target_name].append_value(
                              segment_values.values()
                          )
              targets_dict = self._create_dict_from_segments(targets_segments)
              return targets_segments, targets_dict
      
#### Метод `_create_dict_from_segments` 
Метод `_create_dict_from_segments` преобразует агрегированные сегменты целевого источника данных в словарь,
где ключами являются идентификаторы сегментов целевой переменной, а значениями - словари, содержащие транскрипции агрегированных сегментов
и соответствующие им значения.

          @staticmethod
          def _create_dict_from_segments(
              targets_segments: Dict[str, AggregatedSegment]
          ) -> Dict[str, Dict[str, float]]:
              targets_dict = {}
              for target_segment, aggregated_values in targets_segments.items():
                  targets_dict[target_segment] = {
                      str(value.transcription):
                      float(value.value) if value.value else None
                      for value in aggregated_values.values()
                  }
              return targets_dict

На вход метод принимает словарь следующего типа `targets_segments`:

    {'LNK100_1': {'mill_LNK100_1_Hor1500_ni_0_3_max': mill_LNK100_1_Hor1500_ni_0_3_max, value=443.0,
      'mill_LNK100_1_Hor1500_ni_0_3_min': mill_LNK100_1_Hor1500_ni_0_3_min, value=223.0,
      'mill_LNK100_1_Hor1500_ni_0_3_median': mill_LNK100_1_Hor1500_ni_0_3_median, value=283.0,
      'mill_LNK100_1_Hor2000_ni_0_3_max': mill_LNK100_1_Hor2000_ni_0_3_max, value=719.0,
      ...
      'WBF_sgl_1_Z8_LV2_IN_percent_ni_single': WBF_sgl_1_Z8_LV2_IN_percent_ni_single, value=100.0,
      'WBF_sgl_1_Z9_LV2_IN_percent_ni_single': WBF_sgl_1_Z9_LV2_IN_percent_ni_single, value=100.0,
      'WBF_sgl_1_Z10_LV2_IN_percent_ni_single': WBF_sgl_1_Z10_LV2_IN_percent_ni_single, value=100.0}}

И преобразует его к виду `targets_dict`:

    {'LNK100_1': {'mill_LNK100_1_Hor1500_ni_0_3_max': 443.0,
      'mill_LNK100_1_Hor1500_ni_0_3_min': 223.0,
      'mill_LNK100_1_Hor1500_ni_0_3_median': 283.0,
      'mill_LNK100_1_Hor2000_ni_0_3_max': 719.0,
      ...
      'WBF_sgl_1_Z8_LV2_IN_percent_ni_single': 100.0,
      'WBF_sgl_1_Z9_LV2_IN_percent_ni_single': 100.0,
      'WBF_sgl_1_Z10_LV2_IN_percent_ni_single': 100.0}}

словарь сопоставления агрегированных сегментов сигналов к целевой переменной

---

# 9. Основной обработчик

Этот участок кода создает основной обработчик `Processer`, который будет выполнять основные задачи обработки данных,
используя предоставленные настройки `setup`, метаданные `metadata`, сопоставление данных `mapping`,
настройки источников `source_settings`, фабрику обработчиков `handlers_factory` и сопоставление данных фичей с таргетом `matcher`.

    main_processer = processer.Processer(
        setup=self.setup,
        metadata=metadata,
        data_mapping=mapping,
        sources_settings=sources_settings,
        handlers_factory=handlers_factory,
        matcher=matcher
    )

### Processer Class

      class Processer:
      
          def __init__(
              self,
              handlers_factory: factory.HandlersFactory,
              metadata: pd.DataFrame,
              setup: PipelineSetup,
              data_mapping: dict,
              sources_settings: Dict[str, SourceSettings],
              matcher: Matcher,
          ):
              self.handlers_factory = handlers_factory
              self.matcher = matcher
              self.metadata = metadata
              self.constants = setup
              self.data_mapping = data_mapping
              self.sources_settings = sources_settings
      
          def create_data_batch(self, billet_id: str):
              all_aggregated_sources = {}
              for source, filepath in self.data_mapping[billet_id].items():
                  settings = copy(self.sources_settings[source])
                  data = pd.read_csv(
                      filepath, sep=";", decimal=",", encoding=settings.encoding
                  )
                  source_handler = (
                      self.handlers_factory.get_handler(settings.handler)
                  )
                  all_aggregated_sources[source] = source_handler.process_pipeline(
                      billet_id, data, settings
                  )
              targets_segments, targets_dict = self.matcher.match_features_to_target(
                  all_aggregated_sources
              )
              return targets_segments, targets_dict


Результатом работы объекта класса `Processer` будет словарь, который сопоставляет агрегированные сегменты сигналов с целевой переменной:

    {'LNK100_1': {'mill_LNK100_1_Hor1500_ni_0_3_max': 443.0,
      'mill_LNK100_1_Hor1500_ni_0_3_min': 223.0,
      'mill_LNK100_1_Hor1500_ni_0_3_median': 283.0,
      'mill_LNK100_1_Hor2000_ni_0_3_max': 719.0,
      ...
      'WBF_sgl_1_Z8_LV2_IN_percent_ni_single': 100.0,
      'WBF_sgl_1_Z9_LV2_IN_percent_ni_single': 100.0,
      'WBF_sgl_1_Z10_LV2_IN_percent_ni_single': 100.0}}

Здесь ключ `LNK100_1` представляет собой сегмент целевой переменной, к которому привязаны агрегированные сегменты сигналов.

Значениями для данного ключа является словарь, где ключом является транскрипция сегмента, а значением - его агрегированное значение по данному сегменту.

Тег `workunit` в транскрипции сегмента позволяет определить принадлежность сегмента к целевой переменной или признаку. Транскрипция сегмента имеет следующий формат:

    {workcenter}_{workunit}_{rolling_number}_{name}_{model}_{interpolation}_{sector_range}_{preprocessing}_{aggregation}


---

# 10. Оркестратор

1) Создается список `multiproc_queue`, который будет содержать наборы идентификаторов заготовок.
Каждый набор будет обрабатываться параллельно в отдельном процессе.
2) Размер каждого набора определяется так, чтобы общее количество заготовок было равномерно распределено по
указанному количеству ядер процессора `self.setup.NUM_OF_CORES`.
3) Если общее количество заготовок не делится равномерно на целое число наборов, то некоторые из них могут быть немного
больше, но это не превысит `self.setup.NUM_OF_CORES`.
4) Логируется информация о том, как количество заготовок распределено по ядрам и сколько всего итераций будет выполняться.

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

---

# 11. Запуск агрегации и сохранение метаинформации агрегатора

Этот фрагмент кода отвечает за финальный этап обработки данных и сохранение результатов. Давай подробно разберем, что происходит здесь:

1) Сохранение метаинформации:
- Получаем даты обработки (processing_dates) из метаданных `metadata`, пример показан ниже:


    processing_dates = metadata["Вр.проката"].dt.date


| Вр.проката       | BilletId         | Марка  |
|------------------|------------------|--------|
| 2023-04-01 04:37 | Л22952101_2023   | Э76ХФ  |
| 2023-04-01 04:39 | Л22952105_2023   | Э76ХФ  |
| 2023-04-01 04:41 | Л22952201_2023   | Э76ХФ  |
| 2023-04-01 04:43 | Л22952205_2023   | Э76ХФ  |
| 2023-04-01 04:45 | Л22952301_2023   | Э76ХФ  |
| ...              | ...              | ...    |
| 2023-10-31 07:19 | Л27948204_2023   | 76ХФ   |
| 2023-10-31 07:22 | Л27948301_2023   | 76ХФ   |
| 2023-10-31 07:24 | Л27948401_2023   | 76ХФ   |
| 2023-10-31 07:27 | Л27948304_2023   | 76ХФ   |
| 2023-10-31 07:29 | Л27948404_2023   | 76ХФ   |

Сохраняем первую и последнюю даты обработки в файл `setup.txt` в папке с результатами.
Этот файл содержит общие настройки агрегации, такие как количество ядер процессора, минимальную и максимальную даты обработки, а также другие настройки.

        self._save_settings(
            {
                "FIRST_DATE": str(processing_dates.min()),
                "LAST_DATE": str(processing_dates.max())
            }
        )

#### Метод `_save_settings` класса `MainPipeline`
Метод `_save_settings` создаст файл `setup.txt`, который будет добавлен по пути `PATH_TO_RESULT/LOGS_FOLDER/setup.txt`,
где `PATH_TO_RESULT` - это путь к каталогу для сохранения результатов, а `LOGS_FOLDER` - это подкаталог, указанный в модуле
`constants`, в этом подкаталоге будет сохранен файл `setup.txt`.

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

Файл `setup.txt` может выглядеть следующим образом:

    NUM_OF_CORES = 5
    MIN_TIME = 2023-04-01 00:00:00
    MAX_TIME = 2023-10-31 00:00:00
    MARK_FILTER = True
    MARK = ['Э76ХФ', 'Э90ХАФ', '76ХФ', '90ХАФ']
    METADATA_BILLET_ID = BilletId
    FIRST_DATE = 2023-04-01
    LAST_DATE = 2023-10-31

Сохраняем результаты маппинга по всем заготовкам в CSV-файл report.csv.

        self._save_report(report)

#### Метод `_save_report` класса `MainPipeline`
Метод `_save_report` сохраняет результат маппинга в файл CSV по пути `PATH_TO_RESULT/LOGS_FOLDER/report.csv`,
где `PATH_TO_RESULT` - это путь к каталогу для сохранения результатов, а `LOGS_FOLDER` - это подкаталог, в котором будет сохранен
файл `report.csv`.

    def _save_report(self, report: pd.DataFrame):
        """Сохраняем результаты маппинга по всем заготовкам"""
        report_dir = os.path.join(
            self.setup.PATH_TO_RESULT, constants.LOGS_FOLDER, "report.csv"
        )
        report.to_csv(report_dir, sep=";", decimal=",")

Примерный вид файла `report.csv` будет следующим:

| Time                | BilletId         | LNK100 | U0  | WBF_PIRO | fullness |
|---------------------|------------------|--------|-----|----------|----------|
| 2024-05-01 12:00:00 | Л12345678_2023   | 0      | 1   | 0        | 1/3      |
| 2024-05-02 13:00:00 | Л23456789_2023   | 1      | 0   | 1        | 2/3      |
| ...                 | ...              | ...    | ... | ...      | ...      |


2) Запуск агрегации

- Создаем пул процессов `pool.Pool` для параллельной обработки данных. Количество процессов соответствует числу ядер процессора.
- Используем `tqdm` для отслеживания прогресса обработки данных.
- Для каждого набора заготовок `cut` вызываем метод `create_data_batch` объекта `main_processer` для создания данных.
- Результаты агрегации сохраняются с помощью метода `_save_batch`.

        logger.flog("Запуск аггрегации.", log_type="info")
        for cut_num in tqdm(range(len(multiproc_queue)),
                            bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}'):
            cut = multiproc_queue[cut_num]

            with pool.Pool(self.setup.NUM_OF_CORES) as p:
                batch = p.map(main_processer.create_data_batch, cut)

            self._save_batch(cut, batch)


#### Метод `_save_batch` класса `MainPipeline`

Метод `_save_batch` предназначен для сохранения результатов агрегации. Он принимает список `cut`, содержащий идентификаторы заготовок,
и список `batch`, который содержит результаты агрегации для каждой заготовки.

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

Каждый элемент `batch` представляет собой кортеж, содержащий два словаря, которые сопоставляют агрегированные сегменты сигналов с целевой переменной:

- первый словарь изначальный, и будет сохранен в файл `pkl` по пути `{PATH_TO_RESULT}/{INITIAL_DATA_FOLDER}/{billet_id}.pkl` для каждой заготовки:

  
    {'LNK100_1': {'mill_LNK100_1_Hor1500_ni_0_3_max': mill_LNK100_1_Hor1500_ni_0_3_max, value=443.0,
      'mill_LNK100_1_Hor1500_ni_0_3_min': mill_LNK100_1_Hor1500_ni_0_3_min, value=223.0,
      'mill_LNK100_1_Hor1500_ni_0_3_median': mill_LNK100_1_Hor1500_ni_0_3_median, value=283.0,
      'mill_LNK100_1_Hor2000_ni_0_3_max': mill_LNK100_1_Hor2000_ni_0_3_max, value=719.0,
      ...
      'WBF_sgl_1_Z8_LV2_IN_percent_ni_single': WBF_sgl_1_Z8_LV2_IN_percent_ni_single, value=100.0,
      'WBF_sgl_1_Z9_LV2_IN_percent_ni_single': WBF_sgl_1_Z9_LV2_IN_percent_ni_single, value=100.0,
      'WBF_sgl_1_Z10_LV2_IN_percent_ni_single': WBF_sgl_1_Z10_LV2_IN_percent_ni_single, value=100.0}}

- а второй - обработанный, и будет сохранен в файл `pkl` по пути `{PATH_TO_RESULT}/{PREPARED_DATA_FOLDER}/{target_segment}/{billet_id}.pkl` для каждой заготовки:


    {'LNK100_1': {'mill_LNK100_1_Hor1500_ni_0_3_max': 443.0,
      'mill_LNK100_1_Hor1500_ni_0_3_min': 223.0,
      'mill_LNK100_1_Hor1500_ni_0_3_median': 283.0,
      'mill_LNK100_1_Hor2000_ni_0_3_max': 719.0,
      ...
      'WBF_sgl_1_Z8_LV2_IN_percent_ni_single': 100.0,
      'WBF_sgl_1_Z9_LV2_IN_percent_ni_single': 100.0,
      'WBF_sgl_1_Z10_LV2_IN_percent_ni_single': 100.0}}

Таким образом финальные файлы агрегированных сегментов могут иметь следующий путь: "\\ZSMK-9684-001\Data\DS\aggregation\Prepared_data\LNK100_1\*pkl".

А сам дата-фрейм, собранный из этих `pkl`-файлов выглядеть следующим образом:

| billet_id        | tbk_TBK_1_h_ni_0_3_min | tbk_TBK_1_h_ni_0_3_max | tbk_TBK_1_h_ni_0_3_std | tbk_TBK_1_h_ni_0_3_mean | tbk_TBK_1_h_ni_0_3_median | tbk_TBK_1_h_ni_0_3_tg | tbk_TBK_1_h_ni_0_3_q1 | tbk_TBK_1_h_ni_0_3_q3 | tbk_TBK_1_h_ni_0_3_moda | mill_U0_1_gr232_2.os.table_pos_after_ni_0.75_3_std |
|------------------|------------------------|------------------------|------------------------|-------------------------|---------------------------|-----------------------|-----------------------|-----------------------|-------------------------|----------------------------------------------------|
| Л20001101_2024   | 78.22                  | 181.75                 | 16.177304              | 177.648782              | 180.440                   | 5.372431              | 180.2800              | 180.6400              | 180.29                  | ...                                                |                                                  
| Л20001102_2024   | 163.37                 | 179.95                 | 7.133264               | 172.147025              | 174.930                   | 7.771554              | 163.3700              | 179.7700              | 163.37                  | ...                                                |                                                  
| Л20001103_2024   | 164.03                 | 180.16                 | 6.839057               | 174.206971              | 179.930                   | 7.187111              | 166.2175              | 180.0500              | 164.03                  | ...                                                |                                                  
| Л20001104_2024   | 76.06                  | 182.12                 | 12.944720              | 179.002792              | 180.670                   | 3.209521              | 180.5000              | 180.9100              | 180.91                  | ...                                                |                                                 
| Л20001201_2024   | 160.28                 | 182.73                 | 2.582669               | 180.525459              | 180.670                   | 0.093897              | 180.4000              | 181.0000              | 180.37                  | ...                                                |                                             
| Л20001202_2024   | 164.61                 | 180.23                 | 5.664726               | 175.432452              | 177.550                   | 5.802147              | 174.4075              | 180.0200              | 164.61                  | ...                                                |                           
| Л20001203_2024   | 178.71                 | 181.21                 | 0.348011               | 180.324794              | 180.350                   | -0.304662             | 180.0500              | 180.3900              | 180.37                  | ...                                                |    
| Л20001301_2024   | 159.56                 | 182.11                 | 2.786173               | 180.216166              | 180.550                   | 0.475132              | 180.3000              | 180.7500              | 180.30                  | ...                                                |    
| Л20001302_2024   | 177.94                 | 181.84                 | 0.535102               | 180.481392              | 180.380                   | -0.281646             | 180.2800              | 180.5000              | 180.30                  | ...                                                |  






















