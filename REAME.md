# Video Games JP Sales Prediction (OOP-style Python project)

## Описание
- Проект предсказывает объёмы продаж игр в Японии (JP_Sales) на основе известных продаж в NA, EU и других данных.

## Файлы
- data_load.py — загрузка CSV
- preproc.py — FeatureEngineer (фиксация пропусков, создание признаков)
- model.py — JPRegressor (обёртка вокруг sklearn)
- utils.py — сохранение результата
- main.py — orchestrator: запуск пайплайна

## Логика кода
"""
Запускает весь пайплайн:
1) Загружает данные
2) Применяет FeatureEngineer
3) Делит train->train/val (для валидации)
4) Тренирует JPRegressor
5) Считает MAE на валидации
6) Предсказывает JP_Sales для теста и сохраняет CSV
"""

## Описание файлов 
# data_load.py
"""
Загрузка CSV-файлов  в pandas.DataFrame

Класс DataLoader реализован в 
    Attributes:
        train_path: путь к обучающему файлу (с колонкой JP_Sales)
        test_path: путь к тестовому файлу (без JP_Sales)
        sample_path: пример выходного файла (чтобы взять ID) - не обязателен
"""

# preproc.py
"""
Трансформеры для подготовки признаков. ООП: каждый трансформер реализует fit/transform,
наследует BaseEstimator и TransformerMixin — совместим с sklearn Pipeline/ColumnTransformer.

FeatureEngineer:
- заполняет пропуски
- приводит 'User_Score' к числу (устраняет 'tbd')
- конструирует агрегированную фичу Total_Other_Regions = NA+EU+Other
- удаляет поля Name, Developer, Publisher (чтобы модель не переобучилась на строках)
"""

# model.py
"""
Класс/обёртка JPRegressor:
- строит ColumnTransformer (импутирование/масштабирование)
- предоставляет fit, predict, evaluate (MAE)
- тренирует градиентный бустинг (HistGradientBoostingRegressor) с fallback'ом на RandomForest
Примечание: модель и препроцессинг разделены (ColumnTransformer)
"""

# main.py — Оркестрация констркций 

## Как запускать
1. Поместите `Video_Games.csv`, `Video_Games_Test.csv` и `Video_Games_Sample.csv` в папку `../<project_folder>/data/data` в этом окружении в ту же папку, либо указать пути при вызове `run_pipeline`.
2. Установите зависимости:
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```
3. Запусти `python3 main.py`. Файл с предсказаниями появится по пути `../<project_folder>/data/predictions.csv`

Вывод: В результате будет создан `../<project_folder>/data/predictions.csv` с колонками ID,JP_Sales



