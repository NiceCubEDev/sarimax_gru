# clusterapp-backend

Offline CLI pipeline for Walmart weekly sales forecasting with an honest comparison of
`SARIMAX` and `GRU`.

## What It Does

- loads one store's weekly sales time series from `data/Walmart.csv`
- validates required columns, missing values, duplicate dates, and weekly frequency
- performs a chronological `train / validation / test` split
- trains and selects `SARIMAX` on validation without touching the test split
- trains `GRU` with train-only scaling and early stopping on validation
- compares both models on the test split
- prints a JSON summary and generates a PDF report in `build/reports/`

## Current Architecture

This project is a CLI/offline analytics pipeline, not a FastAPI application.

```text
app/
  cli.py                  command-line entrypoint
  config.py               environment-based pipeline settings
  pipeline/               data loading, validation, splitting, reporting
  analytics/              SARIMAX and GRU model pipelines
  schemas/                Pydantic report/result models
  utils/                  shared metric helpers
data/
  Walmart.csv             input dataset
```

Generated reports and build artifacts are intentionally ignored by git.

## Commands

Use the existing virtual environment:

```bash
make lint
make fmt-check
make initial-methods
make initial-methods STORE_ID=5
```

The main processing command is:

```bash
make initial-methods
```

It prints a JSON summary to stdout and saves the PDF report to `build/reports/`.
