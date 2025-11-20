# Прогнозирование стоимости автомобилей

## Введение
Этот проект посвящён разработке модели для предсказания рыночной стоимости подержанных автомобилей на основе исторических данных о технических характеристиках, комплектациях и ценах. Заказчик — сервис "Не бит, не крашен" — хочет интегрировать модель в приложение для быстрой оценки авто. Приоритеты: качество предсказаний (RMSE), скорость обучения и предсказания.

Проект реализован в Jupyter Notebook (`car_prediction.ipynb`) с использованием Python и библиотек ML.

## Цели и задачи
- **Цель**: Построить модель для определения стоимости автомобиля с учётом ключевых метрик (качество, скорость).
- **Задачи**:
  - Провести анализ данных (EDA): распределения, корреляции, пропуски.
  - Подготовить данные: обработка пропусков, кодирование, масштабирование.
  - Обучить и сравнить модели: LinearRegression, ElasticNet, RandomForestRegressor, LGBMRegressor.
  - Оценить модели по RMSE, времени обучения/предсказания.
  - Выбрать лучшую модель.

## Данные
Датасет: `autos.csv` (354369 строк, 16 столбцов).
- **Признаки**: DateCrawled (дата скачивания), VehicleType (тип кузова), RegistrationYear (год регистрации), Gearbox (коробка передач), Power (мощность), Model (модель), Kilometer (пробег), RegistrationMonth (месяц регистрации), FuelType (топливо), Brand (марка), Repaired (ремонт), DateCreated (дата создания), NumberOfPictures (фото), PostalCode (почтовый индекс), LastSeen (последняя активность).
- **Целевая переменная**: Price (цена в евро).
- Проблемы: Пропуски в VehicleType, Gearbox, Model, FuelType, Repaired; аномалии в годах, мощности, пробеге.

## Реализация
1. **EDA**:
   - Анализ распределений (histograms, boxplots via Matplotlib/Seaborn).
   - Корреляции (phik matrix).
   - Обработка аномалий: удаление выбросов (например, годы <1900 или >2016, мощность >1000 л.с.).
   - Визуализация пропусков (missingno).

2. **Предобработка**:
   - Импутация пропусков (SimpleImputer: mode для категорий, median для чисел).
   - Кодирование: OneHotEncoder/OrdinalEncoder для категорий (VehicleType, Gearbox и т.д.).
   - Масштабирование: StandardScaler/MinMaxScaler для числовых (Power, Kilometer и т.д.).
   - Pipeline с ColumnTransformer для автоматизации.

3. **Модели и обучение**:
   - Разделение данных: train_test_split (80/20).
   - Модели: LinearRegression, ElasticNet, RandomForestRegressor, LGBMRegressor.
   - Гиперпараметры: GridSearchCV/RandomizedSearchCV с CV=5.
   - Метрика: RMSE (neg_root_mean_squared_error).

4. **Оценка**:
   - Кросс-валидация и измерение времени (time module).

## Результаты
- **Сравнение моделей** (на тесте):
  | Модель               | Время обучения (с) | Время предсказания (с) | RMSE    |
  |----------------------|---------------------|-------------------------|---------|
  | LinearRegression     | 4.136              | 0.059                  | 2060.63 |
  | ElasticNet           | 1463.19            | 2.718                  | 2061.25 |
  | RandomForestRegressor| 358.49             | 13.31                  | 1634.83 |
  | LGBMRegressor        | 15.95              | 0.329                  | 1554.96 |

- **Лучшая модель**: LGBMRegressor (низкий RMSE, быстрая скорость).
- Ключевые признаки: RegistrationYear, Power, Kilometer, Brand (высокая корреляция с ценой).
- Вывод: Модель достигает RMSE <1600, что приемлемо для рынка авто. LGBM — оптимальный баланс качества и скорости.

## Используемые инструменты
- Python 3.12+.
- Библиотеки: pandas, numpy, matplotlib, seaborn, missingno, phik, sklearn (model_selection, preprocessing, linear_model, ensemble), lightgbm.
- Требования: `requirements.txt`.

## Как запустить
1. Клонируйте репозиторий: `git clone <repo-url>`.
2. Установите зависимости: `pip install -r requirements.txt`.
3. Запустите Jupyter: `jupyter notebook car_prediction.ipynb`.
4. Данные: Разместите `autos.csv` в `/data/`.