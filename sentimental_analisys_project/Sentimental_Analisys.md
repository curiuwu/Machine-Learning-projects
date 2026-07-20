# Sentimental Analysis Project

Проект для сентиментального анализа отзывов. Основная цель - провести несколько ML-экспериментов, сравнить модели по метрикам, сохранить лучшую модель в виде production-ready артефактов и использовать ее через FastAPI + Streamlit.

Проект поддерживает два типа моделей:

- классический baseline: `TF-IDF + LogisticRegression`;
- sequence-модели на PyTorch: `Word2Vec + RNN`, `Word2Vec + LSTM`.

Текущий лучший кандидат по `macro F1` на test split: `Word2Vec + LSTM`.

## Задача

Модель классифицирует текст отзыва в один из трех классов:

| ID | Класс |
|---:|---|
| 0 | `neutral` |
| 1 | `positive` |
| 2 | `negative` |

Источник данных:

```text
hf://datasets/k1tub/sentiment_dataset/data/train-00000-of-00001.parquet
```

## Сравнение Моделей

Метрики взяты из локальных файлов `artifacts/reports/*/classification_report.json` и `artifacts/models/*/training_config.json`.

| Модель | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Best Val Macro F1 | Best Epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `TF-IDF + LogisticRegression` | 0.7119 | 0.7127 | 0.7118 | 0.7122 | 0.7123 | - | - |
| `Word2Vec + RNN` | 0.6882 | 0.6870 | 0.6881 | 0.6875 | 0.6876 | 0.6794 | 10 |
| `Word2Vec + LSTM` | 0.7173 | 0.7196 | 0.7172 | 0.7175 | 0.7176 | 0.7112 | 10 |

По текущим результатам `LSTM` немного лучше baseline и RNN по test `macro F1`, поэтому именно она выбрана моделью по умолчанию для API.

## Архитектура

```text
sentimental_analisys_project/
  data/
    raw/                       # сырые данные
    processed/                 # обработанные данные

  artifacts/
    models/                    # локальные модели и inference bundles
      logreg/
      rnn/
      lstm/
    plots/                     # графики обучения и confusion matrix
    reports/                   # classification reports и history

  notebooks/                   # исследовательские ноутбуки

  src/
    api/                       # FastAPI service
    dataset/                   # загрузка, очистка, torch Dataset
    embedding/                 # Word2Vec, vocab, embedding matrix
    inference/                 # predictor-ы и factory
    model_training/            # train_logreg, train_rnn, train_lstm
    models/                    # RNNModel, LSTMModel
    plots/                     # графики и classification reports
    streamlit_app/             # Streamlit UI

  docker-compose.yaml
  Dockerfile.app
  Dockerfile.mlflow
  requirements.txt             # полное локальное окружение
  requirements.app.txt         # runtime зависимости для API/Streamlit
```

## Основные Компоненты

`src/dataset/build_dataset.py`
Загрузка parquet-файла из Hugging Face и сохранение raw dataset в `data/raw`.

`src/dataset/clean_dataset.py`
Очистка текста, нормализация `ё -> е`, токенизация и сборка clean dataset.

`src/dataset/torch_dataset.py`
`ReviewsDataset` для sequence-моделей. Возвращает `input_ids`, `length`, `label`.

`src/embedding/`
Построение словаря, обучение Word2Vec и сборка `embedding_matrix`.

`src/models/`
PyTorch-архитектуры `RNNModel` и `LSTMModel`.

`src/model_training/`
Скрипты обучения моделей, логирование в MLflow и сохранение локальных артефактов.

`src/inference/`
Единый интерфейс predictor-ов:

- `BasePredictor`;
- `LogRegPredictor`;
- `SequencePredictor`;
- `factory.create_predictor(...)`.

`src/api/`
FastAPI-приложение с inference endpoints.

`src/streamlit_app/`
Streamlit-интерфейс, который ходит в FastAPI.

## Артефакты Моделей

Локальные артефакты сохраняются в `artifacts/models`.

Для `logreg`:

```text
artifacts/models/logreg/
  tfidf_logreg.pkl
```

Для `rnn` и `lstm`:

```text
artifacts/models/lstm/
  model_state_dict.pt
  embedding_matrix.pt
  checkpoint.pt
  word2idx.pkl
  id2label.json
  label2id.json
  model_config.json
  training_config.json
  preprocessing_config.json
  word2vec/
    word2vec.model
```

Такой bundle нужен для production inference: API может загрузить модель без повторного обучения.

## Установка Локально

Перейти в папку проекта:

```powershell
cd "E:\projects\Проекты по DS\sentimental_analisys_project"
```

Активировать виртуальное окружение:

```powershell
.\.venv\Scripts\activate
```

Установить зависимости:

```powershell
python -m pip install -r requirements.txt
```

Проверить ключевые зависимости:

```powershell
python -c "import torch, mlflow, streamlit, fastapi; print('ok')"
```

## Обучение Моделей

Перед обучением с MLflow нужно поднять MLflow stack:

```powershell
docker compose up -d mlflow postgres minio
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

Запуск baseline:

```powershell
python -m src.model_training.train_logreg
```

Запуск RNN:

```powershell
python -m src.model_training.train_rnn
```

Запуск LSTM:

```powershell
python -m src.model_training.train_lstm
```

Во время обучения:

- raw dataset берется из `data/raw`, если уже существует;
- если raw dataset отсутствует, он скачивается с Hugging Face;
- метрики, параметры, графики и модели логируются в MLflow;
- локальные артефакты сохраняются в `artifacts/models`, `artifacts/plots`, `artifacts/reports`;
- для torch-моделей сохраняется полный inference bundle для FastAPI.

## FastAPI

API использует predictor из `src/inference` и грузит модель один раз при старте приложения.

Доступные endpoints:

| Method | Endpoint | Назначение |
|---|---|---|
| `POST` | `/predict` | Предсказание для одного текста |
| `POST` | `/predict_batch` | Batch inference для списка текстов |
| `GET` | `/info` | Информация о загруженной модели |
| `GET` | `/metrics` | Метрики модели из локальных reports |

Пример запроса:

```json
{
  "text": "Отличный товар, мне понравилось"
}
```

Пример ответа:

```json
{
  "text": "Отличный товар, мне понравилось",
  "tokens": ["отличный", "товар", "мне", "понравилось"],
  "label_id": 1,
  "label": "positive",
  "confidence": 0.93,
  "probabilities": {
    "neutral": 0.04,
    "positive": 0.93,
    "negative": 0.03
  }
}
```

Локальный запуск API:

```powershell
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Документация API:

```text
http://localhost:8000/docs
```

## Streamlit

Streamlit UI не загружает модель напрямую. Он отправляет запросы в FastAPI.

Локальный запуск:

```powershell
streamlit run src/streamlit_app/app.py
```

По умолчанию UI ожидает API на:

```text
http://localhost:8000
```

URL можно изменить через переменную окружения:

```powershell
$env:API_URL="http://localhost:8000"
streamlit run src/streamlit_app/app.py
```

## Docker

Полный запуск сервисов:

```powershell
docker compose up -d --build
```

Сервисы:

| Сервис | URL | Назначение |
|---|---|---|
| MLflow | `http://localhost:5000` | Tracking UI |
| MinIO | `http://localhost:9005` | Artifact storage UI |
| FastAPI | `http://localhost:8000/docs` | Inference API |
| Streamlit | `http://localhost:8501` | UI для пользователя |

По умолчанию API в Docker использует LSTM:

```yaml
MODEL_NAME: lstm
MODEL_ARTIFACT_DIR: /app/artifacts/models/lstm
```

Чтобы переключиться на RNN:

```yaml
MODEL_NAME: rnn
MODEL_ARTIFACT_DIR: /app/artifacts/models/rnn
```

Чтобы переключиться на Logistic Regression:

```yaml
MODEL_NAME: logreg
MODEL_ARTIFACT_DIR: /app/artifacts/models/logreg
```

Остановить сервисы:

```powershell
docker compose down
```

## Docker Dependencies

Для локальной разработки используется полный файл:

```text
requirements.txt
```

Для Docker runtime используется отдельный файл:

```text
requirements.app.txt
```

Это сделано намеренно: полный локальный `requirements.txt` содержит Windows-only зависимости вроде `pywin32`, которые нельзя устанавливать в Linux-контейнер. В `Dockerfile.app` PyTorch ставится как CPU-only wheel, чтобы не тянуть CUDA-зависимости.

## MLflow

Эксперимент:

```text
sentiment_reviews
```

Registered model names:

```text
sentiment-review-classifier
sentiment-review-rnn
sentiment-review-lstm
```

В MLflow логируются:

- параметры обучения;
- train/validation/test метрики;
- графики обучения;
- confusion matrix;
- classification report;
- локальные exports;
- sklearn/PyTorch model artifacts.

## Что Не Попадает В Git

Эти папки считаются локальными рабочими артефактами:

```text
data/
artifacts/
.venv/
```

Они могут быть большими и должны воспроизводиться через загрузку данных, обучение моделей или MLflow artifacts.

## Текущий Статус

Реализовано:

- загрузка и очистка данных;
- обучение `TF-IDF + LogisticRegression`;
- обучение `Word2Vec + RNN`;
- обучение `Word2Vec + LSTM`;
- сохранение локальных inference bundles;
- логирование экспериментов в MLflow;
- FastAPI inference service;
- Streamlit UI;
- Docker Compose для MLflow, PostgreSQL, MinIO, API и Streamlit.

Планируемые улучшения:

- добавить `BiLSTM + Attention`;
- добавить `Transformer BERT/ru-BERT`
- добавить загрузку champion-модели из MLflow через resolver;
- добавить batch upload CSV в Streamlit;
- добавить feedback endpoint для сбора ошибок модели.
