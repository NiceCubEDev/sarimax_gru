# ClusterApp Backend — Документация API

## Обзор

FastAPI-бэкенд для прогнозирования еженедельных продаж сети Walmart.
Реализовано два метода прогнозирования:

| Метод | Подход | Вход | Особенности |
|-------|--------|------|-------------|
| **SARIMAX** | Статистический | Univariate (`Weekly_Sales`) | Тест стационарности, автоподбор параметров, доверительные интервалы |
| **GRU** | Нейросетевой (PyTorch) | Multivariate (6 фичей) | Early stopping, скользящие окна, кривая обучения |

**Базовый URL**: `http://localhost:8000`

---

## Данные

Источник: `data/Walmart.csv` — 45 магазинов, ~143 недели (2010–2012).

| Поле | Тип | Описание |
|------|-----|----------|
| `Store` | int | ID магазина (1–45) |
| `Date` | date | Дата начала недели |
| `Weekly_Sales` | float | Продажи за неделю ($) — **целевая переменная** |
| `Holiday_Flag` | int | Праздничная неделя (0 / 1) |
| `Temperature` | float | Средняя температура (°F) |
| `Fuel_Price` | float | Цена топлива в регионе |
| `CPI` | float | Индекс потребительских цен |
| `Unemployment` | float | Уровень безработицы (%) |

---

## API Роуты

### `GET /health`

Проверка доступности сервера.

**Ответ:**
```json
{"status": "ok"}
```

---

### `GET /api/v1/stores`

Список всех магазинов с агрегированными метриками.

**Параметры:** нет

**Ответ** — массив `StoreAggregate`:
```json
[
  {
    "store_id": 1,
    "mean_sales": 1643735.41,
    "std_sales": 349560.12,
    "mean_temperature": 60.55,
    "mean_cpi": 215.24,
    "mean_unemployment": 7.94,
    "mean_fuel_price": 3.36
  }
]
```

---

### `GET /api/v1/stores/{store_id}/sales`

Все еженедельные продажи конкретного магазина.

**Параметры:**
| Параметр | Тип | Где | Описание |
|----------|-----|-----|----------|
| `store_id` | int | path | ID магазина (1–45) |

**Ответ** — массив `SalesRecord`:
```json
[
  {
    "store": 1,
    "date": "2010-02-05",
    "weekly_sales": 1643690.9,
    "holiday_flag": 0,
    "temperature": 42.31,
    "fuel_price": 2.572,
    "cpi": 211.096,
    "unemployment": 8.106
  }
]
```

---

### `POST /api/v1/forecast/sarimax`

Полный SARIMAX-пайплайн для прогнозирования продаж.

**Тело запроса** — `ForecastRequest`:
```json
{
  "store_id": 1,
  "horizon": 12
}
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `store_id` | int | — | ID магазина |
| `horizon` | int | 12 | Кол-во недель для прогноза в будущее |

**Пайплайн:**
1. ADF-тест стационарности → определение порядка дифференцирования `d`
2. Перебор параметров `(p, d, q)(P, D, Q, s)` по минимальному AIC
3. Train/test split (80/20)
4. Обучение SARIMAX
5. In-sample предсказание (train)
6. Out-of-sample прогноз (test + horizon недель в будущее)
7. Метрики (MAE, RMSE, MAPE) на тестовой выборке
8. Генерация графиков

**Используемые поля данных:** только `Weekly_Sales` (univariate)

**Ответ** — `SarimaxResponse`:
```json
{
  "stationarity": {
    "adf_statistic": -3.4521,
    "p_value": 0.0093,
    "is_stationary": true,
    "differencing_order": 0
  },
  "order": [1, 0, 1],
  "seasonal_order": [1, 0, 1, 52],
  "train_forecast": [
    {"date": "2010-02-05", "actual": 1643690.9, "predicted": 1620000.5, "lower_ci": null, "upper_ci": null}
  ],
  "test_forecast": [
    {"date": "2012-08-03", "actual": 1750000.0, "predicted": 1730000.5, "lower_ci": 1500000.0, "upper_ci": 1960000.0},
    {"date": "2013-01-11", "actual": null, "predicted": 1523000.5, "lower_ci": 1200000.0, "upper_ci": 1846000.0}
  ],
  "metrics": {
    "mae": 95231.45,
    "rmse": 128456.78,
    "mape": 5.67
  },
  "plots": {
    "time_series_url": "/static/plots/timeseries_abc12345.png",
    "forecast_url": "/static/plots/forecast_def67890.png",
    "residuals_url": "/static/plots/residuals_ghi11111.png",
    "acf_pacf_url": "/static/plots/acf_pacf_jkl22222.png",
    "training_loss_url": null
  }
}
```

---

### `POST /api/v1/forecast/gru`

Полный GRU-пайплайн (нейросеть) для прогнозирования продаж.

**Тело запроса** — `ForecastRequest`:
```json
{
  "store_id": 1,
  "horizon": 12
}
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `store_id` | int | — | ID магазина |
| `horizon` | int | 12 | Кол-во недель для прогноза в будущее |

**Пайплайн:**
1. Извлечение 6 фичей из данных магазина
2. MinMaxScaler нормализация
3. Создание скользящих окон длиной `sequence_length` (по умолчанию 12)
4. Train/test split (80/20)
5. Обучение GRU (PyTorch) с early stopping (patience=15)
6. In-sample предсказание (train)
7. Out-of-sample прогноз (test + horizon недель в будущее)
8. Обратное масштабирование предсказаний
9. Метрики (MAE, RMSE, MAPE) на тестовой выборке
10. Генерация графиков

**Используемые поля данных (multivariate):**

| Фича | Описание |
|------|----------|
| `Weekly_Sales` | Целевая переменная + входная фича |
| `Temperature` | Температура |
| `Fuel_Price` | Цена топлива |
| `CPI` | Индекс потребительских цен |
| `Unemployment` | Безработица |
| `Holiday_Flag` | Праздничная неделя |

**Ответ** — `GruResponse`:
```json
{
  "hyperparams": {
    "hidden_size": 64,
    "num_layers": 2,
    "epochs_trained": 85,
    "learning_rate": 0.001,
    "sequence_length": 12,
    "features": ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Holiday_Flag"]
  },
  "train_forecast": [
    {"date": "2010-04-30", "actual": 1643690.9, "predicted": 1620000.5, "lower_ci": null, "upper_ci": null}
  ],
  "test_forecast": [
    {"date": "2012-08-03", "actual": 1750000.0, "predicted": 1734904.1, "lower_ci": null, "upper_ci": null},
    {"date": "2013-01-11", "actual": null, "predicted": 1523000.5, "lower_ci": null, "upper_ci": null}
  ],
  "metrics": {
    "mae": 82100.30,
    "rmse": 110234.50,
    "mape": 4.89
  },
  "plots": {
    "time_series_url": "/static/plots/timeseries_abc12345.png",
    "forecast_url": "/static/plots/forecast_def67890.png",
    "residuals_url": null,
    "acf_pacf_url": null,
    "training_loss_url": "/static/plots/training_loss_xyz99999.png"
  }
}
```

---

## Графики

Графики генерируются серверной стороной (matplotlib) и сохраняются как PNG-файлы.
Доступны по URL из поля `plots` в ответе.

### Общие графики (SARIMAX и GRU)

| График | URL-поле | Оси | Описание |
|--------|----------|-----|----------|
| **Временной ряд** | `time_series_url` | X: Дата, Y: Weekly_Sales | Исходный ряд продаж магазина |
| **Прогноз** | `forecast_url` | X: Дата, Y: Weekly_Sales | 3 линии: Train (факт), Test (факт), Test (прогноз). Для SARIMAX — с 95% доверительным интервалом |

### Только SARIMAX

| График | URL-поле | Оси | Описание |
|--------|----------|-----|----------|
| **Остатки** | `residuals_url` | Два subplot: (1) Остатки по времени, (2) Гистограмма остатков | Визуализация ошибок модели |
| **ACF / PACF** | `acf_pacf_url` | Два subplot: (1) Автокорреляция, (2) Частичная автокорреляция | Помогает оценить структуру ряда |

### Только GRU

| График | URL-поле | Оси | Описание |
|--------|----------|-----|----------|
| **Кривая обучения** | `training_loss_url` | X: Эпоха, Y: Loss (MSE) | Как снижался loss при обучении |

---

## Метрики качества

Обе модели возвращают одинаковые метрики, рассчитанные на **тестовой выборке**:

| Метрика | Формула | Описание |
|---------|---------|----------|
| **MAE** | `mean(\|actual - predicted\|)` | Средняя абсолютная ошибка ($) |
| **RMSE** | `sqrt(mean((actual - predicted)²))` | Корень среднеквадратичной ошибки ($) |
| **MAPE** | `mean(\|actual - predicted\| / actual) × 100` | Средняя ошибка в процентах (%) |

---

## Сравнение методов

| Характеристика | SARIMAX | GRU |
|----------------|---------|-----|
| Тип модели | Статистическая (авторегрессия) | Нейросеть (рекуррентная) |
| Входные данные | `Weekly_Sales` (1 фича) | 6 фичей (multivariate) |
| Доверительные интервалы | ✅ Есть (95%) | ❌ Нет |
| Сезонность | Явная (параметр `s=52`) | Выучивается из данных |
| Интерпретируемость | Высокая (p, d, q) | Низкая (чёрный ящик) |
| Скорость обучения | ~10–30 сек | ~2–5 сек |
| Кривая обучения | Нет | Есть (loss по эпохам) |
| Кэширование | ✅ По (store_id, horizon) | ✅ По (store_id, horizon) |
