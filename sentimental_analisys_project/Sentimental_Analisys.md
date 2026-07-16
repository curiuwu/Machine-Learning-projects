# Sentimental Analysis Project

Проект для сентиментального анализа отзывов. Цель проекта - сравнить несколько подходов к классификации тональности текста, выбрать лучшую модель и подготовить ее к использованию в production-слое через FastAPI и Streamlit.

Текущая базовая production-кандидат модель: `TF-IDF + LogisticRegression`.

## Задача

Модель классифицирует отзыв в один из трех классов:

- `neutral`
- `positive`
- `negative`

Источник данных:

```text
hf://datasets/k1tub/sentiment_dataset/data/train-00000-of-00001.parquet
```

## Текущий статус

Реализовано:

- загрузка raw-датасета из Hugging Face;
- очистка и токенизация текста;
- baseline-обучение `TF-IDF + LogisticRegression`;
- логирование параметров, метрик, графика и модели в MLflow;
- локальное сохранение модели в `artifacts/models`;
- модули для Word2Vec, словарей и embedding matrix;
- `torch Dataset` для RNN/LSTM;
- базовые архитектуры `RNNModel` и `LSTMModel`;
- Docker Compose для MLflow + PostgreSQL + MinIO.

В разработке:

- полноценное обучение RNN/LSTM;
- сохранение полного inference-комплекта для torch-моделей;
- FastAPI inference service;
- Streamlit UI.

## Структура проекта

```text
sentimental_analisys_project/
  data/
    raw/                  # сырые данные
    processed/            # обработанные данные

  artifacts/
    models/               # локально сохраненные модели
    plots/                # графики обучения и оценки
    reports/              # метрики, classification reports, summaries

  notebooks/
    01_logreg.ipynb
    02_word2wec_rnn.ipynb
    03_lstm.ipynb

  scripts/                # тонкие CLI-обертки

  src/
    dataset/              # загрузка, очистка, torch Dataset
    embedding/            # Word2Vec, словари, embedding matrix
    models/               # RNN/LSTM архитектуры
    model_training/       # обучение моделей
    plots/                # код построения графиков

  docker-compose.yaml     # MLflow stack
  Dockerfile.mlflow
  requirements.txt
```

## Установка

Перейти в папку проекта:

```powershell
cd .\sentimental_analisys_project
```

Активировать виртуальное окружение:

```powershell
.\.venv\Scripts\activate
```

Установить зависимости:

```powershell
python -m pip install -r requirements.txt
```

Проверить, что установлен MLflow:

```powershell
python -c "import mlflow; print(mlflow.__version__)"
```

## Запуск MLflow

Поднять MLflow stack:

```powershell
docker compose up -d --build
```

MLflow UI:

```text
http://localhost:5000
```

MinIO UI:

```text
http://localhost:9005
```

Логин и пароль MinIO:

```text
minio
minio123
```

Если bucket `mlflow` не создан автоматически, его нужно создать вручную в MinIO UI.

Остановить сервисы:

```powershell
docker compose down
```

## Обучение Logistic Regression

Запуск baseline-модели:

```powershell
python -m src.model_training.train_logreg
```

Во время запуска:

- если файл `data/raw/sentiment_reviews_raw.parquet` уже есть, он будет использован локально;
- если raw-файла нет, датасет будет скачан с Hugging Face;
- данные будут очищены через `src.dataset.clean_dataset`;
- модель `TF-IDF + LogisticRegression` будет обучена;
- метрики будут залогированы в MLflow;
- learning curve будет сохранен в `artifacts/plots/logreg`;
- модель будет сохранена локально в `artifacts/models/logreg/tfidf_logreg.pkl`;
- модель будет залогирована в MLflow.

## Артефакты

Локальные результаты запусков сохраняются в:

```text
artifacts/
  models/
  plots/
  reports/
```

Эта папка предназначена для локальных результатов экспериментов и не должна попадать в git.

## MLflow

Для эксперимента используется имя:

```text
sentiment_reviews
```

Для модели используется registered model name:

```text
sentiment-review-classifier
```

В MLflow логируются:

- параметры модели;
- validation/test метрики;
- learning curve;
- локальный `.pkl` export;
- sklearn model artifact.

## Основные модули

```text
src/dataset/build_dataset.py
```

Загрузка raw-датасета из Hugging Face и сохранение в `data/raw`.

```text
src/dataset/clean_dataset.py
```

Очистка текста, токенизация и подготовка датасета.

```text
src/dataset/torch_dataset.py
```

`ReviewsDataset` для torch-моделей.

```text
src/embedding/
```

Обучение Word2Vec, построение словарей и embedding matrix.

```text
src/models/
```

Архитектуры `RNNModel` и `LSTMModel`.

```text
src/model_training/train_logreg.py
```

Обучение baseline-модели `TF-IDF + LogisticRegression`.

```text
src/plots/train_curves.py
```

Построение learning curve для sklearn-моделей и training curves для torch-моделей.

## Примечания

- `data/` хранит датасеты.
- `artifacts/` хранит результаты запусков.
- `src/` хранит переиспользуемый код.
- `notebooks/` используются для исследования и экспериментов.
- Перед запуском обучения с MLflow нужно убедиться, что MLflow server доступен на `http://localhost:5000`.
