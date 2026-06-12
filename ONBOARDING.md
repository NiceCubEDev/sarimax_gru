# Онбординг проекта

Проект `clusterapp-backend` - это offline CLI-пайплайн для прогнозирования недельных продаж Walmart по одному магазину. Сейчас это не веб-сервис и не FastAPI-приложение: основной пользовательский сценарий запускается из командной строки, обучает две модели (`SARIMAX` и `GRU`), сравнивает их на тестовом периоде и сохраняет PDF-отчет.

## Быстрый старт

Работать удобнее из WSL/Linux, потому что `makefile` использует путь `.venv/bin/python`.

```bash
cd /home/salavat/agni/sarimax_gru
uv sync --dev
cp .env.example .env
mkdir -p data
# положите датасет в data/Walmart.csv
make lint
make fmt-check
make initial-methods STORE_ID=1 HORIZON=12
```

Результат выполнения `make initial-methods`:

- JSON-сводка печатается в stdout.
- PDF-отчет сохраняется в `build/reports/store_<STORE_ID>_forecast_comparison.pdf`.

Если `make` недоступен, можно запустить CLI напрямую:

```bash
.venv/bin/python -m app.cli initial-methods --store-id 1 --horizon 12
```

## Что нужно перед запуском

- Python `>=3.11`.
- `uv` для установки зависимостей из `pyproject.toml` и `uv.lock`.
- Файл `data/Walmart.csv`. CSV не хранится в git: `.gitignore` исключает `*.csv`.
- При необходимости настройки можно переопределить через `.env`, шаблон лежит в `.env.example`.

Ожидаемые колонки датасета:

- `Store`
- `Date`
- `Weekly_Sales`
- `Temperature`
- `Fuel_Price`
- `CPI`
- `Unemployment`
- `Holiday_Flag`

`Date` читается с `dayfirst=True`, поэтому формат вроде `05-02-2010` будет интерпретирован как `5 февраля 2010`.

## Структура проекта

```text
app/
  cli.py                    CLI-точка входа
  config.py                 настройки через pydantic-settings и .env
  analytics/
    sarimax.py              обучение, подбор и прогноз SARIMAX
    gru.py                  обучение, тестирование и рекурсивный прогноз GRU
  pipeline/
    data_loader.py          загрузка CSV и выбор магазина
    forecasting.py          проверка ряда, split, будущие признаки
    reporting.py            сбор общего отчета и генерация PDF
  schemas/
    forecast.py             Pydantic-схемы результата
  utils/
    metrics.py              MAE, RMSE, MAPE, sMAPE, WAPE
```

Корневые файлы:

- `pyproject.toml` - зависимости, сборка пакета, настройки Ruff.
- `uv.lock` - зафиксированные версии зависимостей.
- `makefile` - основные команды разработки и запуска.
- `.env.example` - шаблон переменных окружения.
- `.gitignore` - игнорирует данные, окружение и build-артефакты.

## Основной поток данных

1. `app.cli` разбирает команду `initial-methods`, `--store-id` и `--horizon`.
2. `data_loader.get_store_data()` читает `data/Walmart.csv`, фильтрует один магазин и делает `Date` индексом.
3. `validate_store_dataframe()` проверяет обязательные колонки, пропуски, дубликаты дат, конечность чисел и регулярную недельную частоту.
4. `split_time_series()` делает хронологическое разбиение `train / test` в пропорции 80% / 20%.
5. `run_sarimax_pipeline()`:
   - проверяет стационарность ADF-тестом;
   - подбирает параметры по AIC на обучающей части с внешними признаками;
   - обучает модель на `train`;
   - оценивает качество на `test`;
   - отдельно обучается на всей истории и использует прогнозные будущие признаки для будущего прогноза.
6. `run_gru_pipeline()`:
   - добавляет календарные признаки `Week_Sin` и `Week_Cos`;
   - масштабирует признаки только по train для честной оценки;
   - обучает GRU на `train`;
   - оценивает качество на `test`;
   - строит рекурсивный будущий прогноз.
7. `generate_comparison_report()` собирает Pydantic-структуру `ComparisonReport`, печатает JSON через CLI и генерирует PDF.

## Настройки

Настройки живут в `app/config.py` и могут быть заданы через `.env`.

Чаще всего меняются:

- `CSV_PATH` - путь к датасету, по умолчанию `data/Walmart.csv`.
- `REPORTS_DIR` - директория PDF-отчетов, по умолчанию `build/reports`.
- `TRAIN_RATIO`, `TEST_RATIO` - доли хронологического разбиения, сумма должна быть `1.0`.
- `FORECAST_HORIZON` - горизонт будущего прогноза по умолчанию.
- `SARIMAX_SEASONAL_PERIOD` - сезонность SARIMAX, сейчас `52` недели.
- `GRU_SEQUENCE_LENGTH` - длина окна GRU, сейчас `12`.
- `GRU_EPOCHS`, `GRU_HIDDEN_SIZE`, `GRU_NUM_LAYERS`, `GRU_LEARNING_RATE` - параметры обучения GRU.

## Модели и честность оценки

В проекте явно заложена идея "честного" сравнения моделей:

- Первые 80% истории используются для обучения, последние 20% - для тестирования.
- SARIMAX подбирает `order` и `seasonal_order` по AIC только на обучающей части.
- GRU обучается только на train; тест не участвует в обучении.
- Скейлеры GRU для оценки fit'ятся только на train.
- Будущий прогноз строится уже отдельными моделями, обученными на всей доступной истории.

Важно: будущие погодные и экономические признаки для GRU (`Temperature`, `Fuel_Price`, `CPI`, `Unemployment`) сначала прогнозируются отдельными временными моделями, а `Holiday_Flag` вычисляется по известным праздничным неделям Walmart.

## Отчеты

PDF состоит из нескольких страниц:

- обзор данных, split и выбранных моделей;
- сравнение факта и прогнозов на тестовом периоде;
- история, модельные линии и будущий прогноз;
- абсолютная ошибка на тесте и loss GRU.

Все build-артефакты игнорируются git'ом.

## Команды разработки

```bash
make lint
make lint-fix
make fmt
make fmt-check
make initial-methods STORE_ID=1 HORIZON=12
make clean
```

Тестов в текущем репозитории нет, хотя `pytest` подключен в dev-зависимостях. При изменениях в логике split, метрик или моделей стоит добавить unit-тесты вокруг этих функций.

## Как вносить изменения

Если меняется формат входных данных:

- обновить `REQUIRED_COLUMNS` и/или `CONTINUOUS_FUTURE_COLUMNS` в `app/pipeline/forecasting.py`;
- проверить `FEATURE_COLUMNS` в `app/analytics/gru.py`;
- обновить этот онбординг и README.

Если добавляется новая модель:

- добавить модуль в `app/analytics/`;
- добавить Pydantic-схемы результата в `app/schemas/forecast.py`;
- подключить модель в `generate_comparison_report()`;
- добавить графики и строки метрик в `app/pipeline/reporting.py`;
- расширить CLI/README, если появляются новые параметры запуска.

Если меняются метрики:

- править `app/utils/metrics.py`;
- убедиться, что `ModelMetrics` в `app/schemas/forecast.py` соответствует новым полям;
- обновить таблицу метрик в PDF.

## Известные особенности и риски

- В текущей рабочей копии нет `data/Walmart.csv`, поэтому полный пайплайн не запустится без внешнего датасета.
- Нет `.venv`, окружение нужно поднять через `uv sync --dev`.
- `makefile` рассчитан на Linux/WSL-путь `.venv/bin/python`; из Windows PowerShell команды `make` могут работать не так, как ожидается.
- Репозиторий называется backend, но HTTP API сейчас отсутствует.
- Подбор SARIMAX и обучение GRU могут занимать заметное время на CPU.
- Полного тестового покрытия нет.
