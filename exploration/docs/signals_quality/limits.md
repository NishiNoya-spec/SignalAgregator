# Статистический расчет лимитов сигналов
Попадание значения сигнала в заданные пределы - один из критериев качества сигналов

## Инструкция:

1) [Препроцессинг](../preprocessing.md) данных
2) Запуск exploration/signals_quality/limmits/limmits.py:
   1) USE_COLUMNS. Названия сигналов для установки лимитов
   2) PATH_TO_PREPROCESSED_DATA. Путь до предобработанных данных
   3) SAVE_PATH. Путь для сохранения результатов работы алгоритма.
3) Проанализировать полученные диаграммы и определенные лимиты. При необходимости скорректировать.
4) Внести изменения лимитов в файлы по пути components/backend_tpqc/src/tpqc/adapters/signals/limits