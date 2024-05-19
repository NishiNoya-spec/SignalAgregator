# Общая информация
В рамках исследований решаются задачи:

1) Разработка моделей предсказания дефектов на рельсе:
   1) [скрученность](models/torsion.md);
   2) [неметаллические включения](models/nonmetalic_muzk.md);

2) Определение порогов значений сигналов

## Структура exploration

- docs: вся информация об исследованиях и инструкции к ним. (Формируется согласно внутренней структуре exploration)
- models: все эксперименты, проводимые в рамках проекта. (Новый вид дефекта или подход - новая директория)
- special_zones_research: исследования новых зон агрегаций.
- signals_quality: исследования в рамках контроля качества сигналов. (Новый подход - новая директория)
- utils: универсальные методы и классы
- heap: временное хранение потенциально неактуальных файлов. (Должны быть либо удалены, либо восстановлены согласно структуре exploration)

## Начало работы

### Настройка среды разработки

1) Установка окружения:


    cd ~/evraz/9684/9684/exploration
    rm -rf venv && python3.10 -m venv venv
    
2) Добавление в venv pip.conf (pip.ini для Windows):


    [global]
    trusted-host = sib-reg-001.sib.evraz.com
    index = https://sib-reg-001.sib.evraz.com/repository/pypi-group/pypi
    index-url = https://sib-reg-001.sib.evraz.com/repository/pypi-group/simple
    extra-index-url = https://sib-reg-001.sib.evraz.com/repository/sib-pypi/pypi

3) Активация среды:
   

    source venv/bin/activate && python --version && pip install --upgrade pip

4) Установка python пакетов:
   

    pip install -e .

### Сделать перед отправкой в ПР

      cd ~/evraz/9684/9684/exploration
      clear && isort && yapf
      clear && flake8


