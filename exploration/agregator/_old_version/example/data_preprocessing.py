import math
import os
from multiprocessing import cpu_count, pool

import pandas as pd
from tqdm import tqdm
from datetime import datetime

from exploration.experiments_aggregation.utils import create_save_path
from exploration.experiments_aggregation import data_mapping, dto, processer
from exploration.experiments_aggregation.logger import logger
from exploration.experiments_aggregation.source_parser import parse_settings


constants = dto.Constants(
    PATH_TO_RESULT=r"S:\DS\test_model_data",
    NUM_OF_CORES=5,
    MIN_TIME=datetime(year=2023, month=11, day=10, hour=0, minute=0, second=0),
    MAX_TIME=datetime(year=2023, month=11, day=12, hour=0, minute=0, second=0),
    MARK_FILTER=True,
    MARK=['Э76ХФ', 'Э90ХАФ', '76ХФ', '90ХАФ'],
    PATH_TO_METADATA=os.path.join(r'S:\DS',
                                  "metadata/*xlsx"),
    METADATA_BILLET_ID="BilletId"
)


def preprocessing():
    settings = pd.read_excel(
        "settings.xlsx", index_col=[0], header=None).iloc[:, :-1]
    sources_settings = parse_settings(settings)

    logger.info("Подготовка данных. Создание мапинга...")
    print("Подготовка данных. Создание мапинга...")

    creator = data_mapping.MappingCreator(constants=constants)
    metadata = creator.get_metadata()

    processing_dates = metadata["Вр.проката"].dt.date
    result_folder_name = str(processing_dates.min()) + "_" + str(processing_dates.max())

    mapping, report = creator.create_mapping(metadata=metadata,
                                             settings=sources_settings,
                                             time_period=result_folder_name)

    main_processer = processer.Processer(constants=constants,
                                         metadata=metadata,
                                         data_mapping=mapping,
                                         sources_settings=sources_settings)

    result_dir = create_save_path(constants.PATH_TO_RESULT, result_folder_name)
    # Оркестратор
    multiproc_queue = []
    for n in range(math.ceil(len(mapping) / constants.NUM_OF_CORES)):
        multiproc_queue.append(
            list(mapping.keys())[(constants.NUM_OF_CORES *
                                  n):(constants.NUM_OF_CORES * (n + 1))])

    print("Мапинг создан. Заготовки распределены по процессам")
    logger.info(
        f"{len(mapping)} заготовок распределены по {constants.NUM_OF_CORES}"
        f"процессам. Всего {len(multiproc_queue)} итераций")

    print("Запуск аггрегации...")
    for cut_num in tqdm(range(len(multiproc_queue)),
                        bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}'):

        cut = multiproc_queue[cut_num]
        # Debug
        deb = main_processer.create_data_batch(cut[1])

        with pool.Pool(constants.NUM_OF_CORES) as p:
            batch = p.map(main_processer.create_data_batch, cut)

        logger.info(
            f"Закончена аггрегация {len(batch)} файлов пакета {cut_num} из "
            f"{len(multiproc_queue)}"
        )

        if not os.path.exists(constants.PATH_TO_RESULT):
            os.makedirs(constants.PATH_TO_RESULT)
        batches = []
        for billet_batch in batch:
            if billet_batch is not None:
                batches.extend(billet_batch)
        del batch

        df = pd.DataFrame(batches)
        save_path = os.path.join(result_dir, f'{cut_num}.csv')
        df.to_csv(save_path)
        logger.info(
            f"Данные успешно сохранены по пути:  {save_path} \n"
        )


if __name__ == "__main__":
    preprocessing()
