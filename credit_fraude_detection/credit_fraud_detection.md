# Credit Fraud Detection Project

Проект по обнаружению мошеннических транзакций с использованием машинного обучения. Включает полный анализ данных, обучение нескольких моделей и интерпретацию результатов с помощью SHAP.

[Ознакомиться с кодом можно тут:](https://github.com/curiuwu/Machine-Learning-projects/blob/main/credit_fraude_detection/credit_fraud_detection.ipynb)

## Описание

Этот проект демонстрирует полный пайплайн машинного обучения для задачи бинарной классификации: обнаружение мошеннических кредитных транзакций. Проект включает:

- Разведывательный анализ данных (EDA)
- Обработка дисбаланса классов с помощью SMOTE
- Обучение и сравнение 4 моделей: Logistic Regression, Decision Tree, Random Forest, LightGBM
- Оценка моделей с использованием метрик precision, recall, F1-score, ROC-AUC, PR-AUC
- Интерпретация модели с помощью SHAP (SHapley Additive exPlanations)

Итоговая модель: **LightGBM** - выбрана за оптимальный баланс точности и скорости обучения.

## Датасет

Используется датасет **Credit Card Fraud Detection** с Kaggle.

- **Ссылка**: [https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Описание**: Датасет содержит транзакции, совершенные европейскими держателями карт в сентябре 2013 года. Включает 284,807 транзакций, из которых 492 (0.172%) являются мошенническими.
- **Признаки**: 28 анонимизированных признаков (V1-V28), полученных с помощью PCA, плюс Time и Amount.
- **Целевая переменная**: Class (0 - легальная транзакция, 1 - мошенническая).

Скачайте датасет и поместите файл `creditcard.csv` в папку `../data/` относительно корня проекта.

## Установка и быстрый запуск

### Предварительные требования

- Python 3.8+
- Jupyter Notebook или JupyterLab
- Git (для клонирования репозитория)

### Установка зависимостей

1. Клонируйте репозиторий:
```bash
git clone https://github.com/your-username/Machine-Learning-projects.git
cd Machine-Learning-projects/credit_fraude_detection
```

2. Установите зависимости:
```bash
pip install -r ../../requirements.txt
```

Или установите основные библиотеки вручную:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn lightgbm shap jupyter
```

### Запуск проекта

1. Скачайте датасет с Kaggle и поместите `creditcard.csv` в папку `../data/`

2. Запустите Jupyter Notebook:
```bash
jupyter notebook credit_fraud_detection.ipynb
```

3. Выполните ячейки последовательно. Проект настроен для автоматического выполнения с RANDOM_STATE=42 для воспроизводимости.

### Быстрый запуск (без установки)

Если у вас установлен Google Colab:
1. Загрузите ноутбук `credit_fraud_detection.ipynb` в Colab
2. Скачайте датасет и загрузите его в Colab
3. Измените путь к файлу в первой ячейке на соответствующий
4. Запустите все ячейки

## Структура проекта

```
credit_fraude_detection/
├── credit_fraud_detection.ipynb    # Основной ноутбук с анализом и обучением
├── credit_fraud_detection.md       # Описание проекта
data/
└── creditcard.csv                  # Датасет (скачать с Kaggle)

requirements.txt                    # Зависимости проекта
```

## Результаты

### Сравнение моделей (CV Average Precision)

| Модель                  | Время обучения (сек) | Время предсказания (сек) | Average Precision |
|-------------------------|----------------------|---------------------------|-------------------|
| Логистическая регрессия | 0.217               | 0.008                    | 0.747            |
| Дерево решений         | 3.675               | 0.018                    | 0.646            |
| Случайный лес          | 256.888             | 0.845                    | 0.838            |
| **LightGBM**           | **26.902**          | **0.489**                | **0.822**        |

### Метрики на тестовом наборе (LightGBM)

- F1-score: 0.846
- ROC-AUC: 0.923
- PR-AUC: 0.846

### Ключевые выводы

- LightGBM показал лучший баланс между точностью и скоростью
- SHAP анализ показал, что признаки V4, V12, V14 наиболее важны для предсказания
- Модель эффективно обрабатывает дисбаланс классов благодаря class_weight='balanced'

## Использование модели

После выполнения ноутбука лучшая модель доступна как `best_model`. Для предсказания на новых данных:

```python
# Пример использования
new_transaction = X_test.iloc[0:1]  # Новые данные
prediction = best_model.predict(new_transaction)
probability = best_model.predict_proba(new_transaction)[:, 1]
```
