# Предсказание Дефолта по Кредитным Картам - MLOps Проект

Сквозной (end-to-end) MLOps пайплайн для предсказания вероятности дефолта по кредитным картам с использованием машинного обучения.

## Структура проекта

```
credit-card-default-prediction/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI пайплайн
├── data/
│   ├── raw/                    # Сырые данные (не отслеживаются в git)
│   └── processed/              # Обработанные данные (отслеживаются DVC)
├── models/                     # Обученные модели (отслеживаются DVC)
├── notebooks/                  # Jupyter notebooks для EDA
├── reports/
│   └── figures/                # Сгенерированные графики
├── scripts/
│   ├── docker.sh               # Скрипт сборки и запуска Docker
│   ├── monitor.py              # Мониторинг и детекция дрифта
│   ├── run_experiments.py      # Запуск множественных экспериментов
│   └── train_model.py          # Скрипт обучения для DVC пайплайна
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI приложение
│   ├── data/
│   │   ├── load_data.py        # Утилиты загрузки данных
│   │   ├── prepare.py          # Пайплайн подготовки данных
│   │   └── validate.py         # Валидация данных с Pandera
│   ├── features/
│   │   └── engineer.py         # Feature engineering
│   └── models/
│       └── train.py            # Обучение модели с MLflow
├── tests/
│   └── test_pipeline.py        # Unit-тесты
├── docker-compose.yml          # Docker Compose конфигурация
├── Dockerfile                  # Определение Docker образа
├── dvc.yaml                    # Определение DVC пайплайна
├── params.yaml                 # DVC параметры
├── requirements.txt            # Python зависимости
├── Makefile                    # Make команды
└── README.md                   # Этот файл
```

## Установка

### Предварительные требования

- Python 3.10+
- Git
- Docker (опционально, для контейнеризации)
- Kaggle API credentials (для загрузки данных)

### Настройка

1. **Клонирование репозитория**:
```bash
git clone https://github.com/lobasfs/default_of_credit_card
cd default_of_credit_card
```

2. **Создание виртуального окружения**:
```bash
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

3. **Установка зависимостей**:
```bash
pip install -r requirements.txt
```

4. **Настройка Kaggle API** (для загрузки данных):
```bash
# Поместите ваш kaggle.json в ~/.kaggle/
mkdir -p ~/.kaggle
cp /path/to/your/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

5. **Инициализация DVC**:
```bash
dvc init
```

## Быстрый старт

### 1. Загрузка данных

```bash
# Вариант 1: Использование Kaggle API
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset -p data/raw --unzip

# Вариант 2: Ручная загрузка
# Скачайте с https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
# Поместите UCI_Credit_Card.csv в data/raw/
```

### 2. Запуск полного пайплайна

```bash
# Запуск DVC пайплайна (подготовка данных + обучение модели)
dvc repro

# Или запуск этапов по отдельности:
python -m src.data.prepare
python scripts/train_model.py
```

### 3. Просмотр MLflow экспериментов

```bash
mlflow ui --port 5000
# Откройте http://localhost:5000 в браузере
```

### 4. Запуск API

```bash
# Вариант 1: Прямой запуск Python
python -m src.api.app

# Вариант 2: Использование Docker Compose
docker-compose up -d

# Вариант 3: Использование Docker скрипта
./scripts/docker.sh compose-up
```

API будет доступно по адресам:
- API: http://localhost:8000
- API Документация: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

## Использование

### Подготовка данных

```python
from src.data.prepare import prepare_data

# Подготовка данных с пользовательскими параметрами
train_df, test_df = prepare_data(
    input_path="data/raw/UCI_Credit_Card.csv",
    output_dir="data/processed",
    test_size=0.2,
    random_state=42
)
```

### Обучение модели

```python
from src.models.train import train_model

# Обучение модели
pipeline = train_model(
    model_type="logistic_regression",
    tune_hyperparameters=True
)
```

### Запуск множественных экспериментов

```bash
python scripts/run_experiments.py
```

Это обучит 6 различных моделей:
- Logistic Regression (по умолчанию и с подбором гиперпараметров)
- Random Forest (по умолчанию и с подбором гиперпараметров)
- Gradient Boosting (по умолчанию и с подбором гиперпараметров)

### Выполнение предсказаний через API

```bash
# Одиночное предсказание
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 20000.0,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -2,
    "PAY_6": -2,
    "BILL_AMT1": 3913.0,
    "BILL_AMT2": 3102.0,
    "BILL_AMT3": 689.0,
    "BILL_AMT4": 0.0,
    "BILL_AMT5": 0.0,
    "BILL_AMT6": 0.0,
    "PAY_AMT1": 0.0,
    "PAY_AMT2": 689.0,
    "PAY_AMT3": 0.0,
    "PAY_AMT4": 0.0,
    "PAY_AMT5": 0.0,
    "PAY_AMT6": 0.0
  }'
```

Ответ:
```json
{
  "prediction": 1,
  "probability": 0.753,
  "risk_level": "high"
}
```

## MLOps компоненты

### 1. Валидация данных (Pandera)

Валидация схемы данных обеспечивает качество данных:

```python
from src.data.validate import validate_data

validated_df = validate_data(df)
```

Проверяет:
- Типы данных
- Диапазоны значений
- Обязательные поля
- Бизнес-правила

### 2. Feature Engineering

Автоматическое создание признаков:

```python
from src.features.engineer import engineer_features

df_engineered = engineer_features(df)
```

Создает:
- Агрегации поведения платежей
- Тренды сумм счетов
- Коэффициенты платежей
- Использование кредита
- Группы возраста и кредитного лимита

### 3. Отслеживание экспериментов (MLflow)

Автоматическое отслеживание всех экспериментов:

```python
with mlflow.start_run():
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_metric("test_roc_auc", 0.75)
    mlflow.sklearn.log_model(model, "model")
```

Логируемые артефакты:
- Параметры
- Метрики (ROC-AUC, Precision, Recall, F1)
- Модели
- Графики (ROC-кривая, матрица ошибок)

### 4. Контроль версий (DVC)

Отслеживание данных и моделей:

```bash
# Добавление данных в DVC
dvc add data/raw/UCI_Credit_Card.csv

# Запуск пайплайна
dvc repro

# Отправка в удаленное хранилище
dvc push
```

### 5. CI/CD (GitHub Actions)

Автоматизированное тестирование при каждом push:
- Линтинг кода (flake8)
- Форматирование кода (black)
- Unit-тесты (pytest)
- Валидация данных
- Отчеты о покрытии

### 6. Контейнеризация (Docker)

```bash
# Сборка образа
docker build -t credit-card-api .

# Запуск контейнера
docker run -p 8000:8000 credit-card-api

# Или использование Docker Compose
docker-compose up
```

### 7. Мониторинг (Дрифт данных)

Мониторинг дрифта данных:

```bash
python scripts/monitor.py
```

Рассчитывает PSI (Population Stability Index) для:
- Отдельных признаков
- Предсказаний модели

Пороги PSI:
- < 0.1: Нет значительного дрифта
- 0.1 - 0.2: Умеренный дрифт
- \> 0.2: Значительный дрифт (рекомендуется переобучение)

## API документация

### Endpoints

#### `GET /`
Корневой endpoint с информацией об API.

#### `GET /health`
Проверка работоспособности.

Ответ:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### `POST /predict`
Выполнение предсказания для одного экземпляра.

Тело запроса: См. схему PredictionInput в документации API.

Ответ:
```json
{
  "prediction": 1,
  "probability": 0.753,
  "risk_level": "high"
}
```

#### `POST /batch_predict`
Выполнение предсказаний для нескольких экземпляров.

### Интерактивная документация API

FastAPI предоставляет автоматическую интерактивную документацию:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Тестирование

### Запуск всех тестов

```bash
pytest tests/ -v
```

### Запуск тестов с покрытием

```bash
pytest tests/ -v --cov=src --cov-report=html
```

### Запуск конкретного теста

```bash
pytest tests/test_pipeline.py::TestFeatureEngineering -v
```

### Категории тестов

- **Тесты Feature Engineering**: Проверка создания признаков
- **Тесты валидации данных**: Проверка валидации схемы
- **Тесты обучения модели**: Тестирование компонентов модели
- **Тесты целостности данных**: Проверка качества данных

## Развертывание

### Локальное развертывание

```bash
# Использование Docker
./scripts/docker.sh build
./scripts/docker.sh run

# Использование Docker Compose
./scripts/docker.sh compose-up
```

## Мониторинг

### Мониторинг дрифта данных

```bash
# Запуск скрипта мониторинга
python scripts/monitor.py
```

Мониторит:
- Изменения распределения признаков
- Изменения распределения предсказаний
- Метрики производительности модели

## Использование Makefile

Для удобства используйте команды Makefile:

```bash
make help           # Показать все команды
make install        # Установить зависимости
make data           # Подготовить данные
make train          # Обучить модель
make api            # Запустить API
make test           # Запустить тесты
make docker-build   # Собрать Docker образ
make monitor        # Запустить мониторинг
make dvc-repro      # Запустить DVC пайплайн
```
