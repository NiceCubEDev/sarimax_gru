# clusterapp-backend

Offline pipeline for Walmart weekly sales forecasting with honest comparison of `SARIMAX` and `GRU`.

## What it does

- validates the input store time series
- performs a chronological `train / validation / test` split
- trains and selects `SARIMAX` on validation without touching the test split
- trains `GRU` with train-only scaling and early stopping on validation
- compares both models on the test split
- generates a single PDF report in `build/reports/`

## Commands

Use the existing virtual environment:

```bash
make lint
make initial-methods
make initial-methods STORE_ID=5
```

The main processing command is:

```bash
make initial-methods
```

It prints JSON summary to stdout and saves the PDF report to `build/reports/`.
