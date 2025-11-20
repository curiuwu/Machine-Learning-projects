# Кредитный скоринг

## Введение
Проект фокусируется на разработке модели для кредитного скоринга: предсказание кредитного рейтинга клиента ("Good", "Standard", "Poor") на основе личных и финансовых данных. Модель предназначена для API интеграции. Приоритеты: высокая скорость обучения/предсказания, метрика recall (F2-score с beta=2 для акцента на false negatives).

Реализовано в Jupyter Notebook (`credit_scoring.ipynb`) с использованием Python и ML-библиотек.

## Цели и задачи
- **Цель**: Обучить модель для классификации кредитного риска с акцентом на recall.
- **Задачи**:
  - EDA: распределения, корреляции, пропуски.
  - Предобработка: очистка, кодирование, импутация.
  - Обучить модели: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, LGBMClassifier, CatBoostClassifier.
  - Оценить по F2-score, времени.
  - Сохранить лучшую модель (joblib).

## Данные
Датасет: `credit_scoring_data.csv` (100000 строк, 28 столбцов).
- **Признаки**: id, customer_id, month, name, age, ssn, occupation, annual_income, monthly_inhand_salary, num_bank_accounts, num_credit_card, interest_rate, num_of_loan, type_of_loan, delay_from_due_date, num_of_delayed_payment, changed_credit_limit, num_credit_inquiries, credit_mix, outstanding_debt, credit_utilization_ratio, credit_history_age, payment_of_min_amount, total_emi_per_month, amount_invested_monthly, payment_behaviour, monthly_balance.
- **Целевая переменная**: credit_score (мультикласс: "Good", "Standard", "Poor").
- Проблемы: Смешанные типы, пропуски, аномалии (например, отрицательные значения, строки с "_").

## Реализация
1. **EDA**:
   - Распределения (histograms, boxplots).
   - Корреляции (phik).
   - Визуализация пропусков (missingno).
   - Обработка: Удаление дубликатов, исправление аномалий (например, age >100 → NaN).

2. **Предобработка**:
   - Импутация: SimpleImputer (median/mode).
   - Кодирование: OneHotEncoder/OrdinalEncoder для категорий (occupation, credit_mix и т.д.).
   - Масштабирование: MinMaxScaler для числовых.
   - Pipeline с ColumnTransformer.
   - Балансировка классов (если нужно).

3. **Модели и обучение**:
   - Разделение: train_test_split (stratified, 80/20).
   - Модели: LogisticRegression, DecisionTree, RandomForest, LGBM, CatBoost.
   - Гиперпараметры: GridSearchCV с CV=5, scorer=F2.
   - Метрика: F2-score (beta=2).

4. **Оценка**:
   - Кросс-валидация, classification_report, время.

## Результаты
- **Сравнение моделей** (на тесте):
  | Модель                | Время обучения (с) | Время предсказания (с) | F2-score |
  |-----------------------|---------------------|-------------------------|----------|
  | LogisticRegression    | 17.65              | 0.093                  | 0.852   |
  | DecisionTreeClassifier| 0.423              | 0.095                  | 0.896   |
  | RandomForestClassifier| 46.28              | 1.47                   | 0.889   |
  | LGBMClassifier        | 1.28               | 0.167                  | 0.883   |

- **Лучшая модель**: LGBMClassifier (F2=0.883 на тесте, быстрая, устойчивая к переобучению).
- Ключевые признаки: outstanding_debt, interest_rate, delay_from_due_date (высокая корреляция с риском).
- Вывод: Модель достигает F2 >0.88, минимизируя риски false negatives. Сохранена как `LGBM_model.joblib`.

## Используемые инструменты
- Python 3.13+.
- Библиотеки: pandas, numpy, matplotlib, seaborn, missingno, phik, sklearn, lightgbm, catboost, joblib.
- Требования: `requirements.txt`.

## Как запустить
1. Клонируйте репозиторий: `git clone <repo-url>`.
2. Установите зависимости: `pip install -r requirements.txt`.
3. Запустите Jupyter: `jupyter notebook credit_scoring.ipynb`.
4. Данные: Разместите `credit_scoring_data.csv` в `/data/`.
5. Загрузка модели: `joblib.load('LGBM_model')`.