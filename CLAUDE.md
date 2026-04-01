# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for package management with Python 3.13.

```bash
# Install dependencies
uv sync

# Run any script
uv run python main.py
uv run python global_model_lstm.py
uv run python test.py
uv run python dataset_preparation.py
uv run python prediction.py --product_id P0003 --epochs 20
```

## Project Overview

A time series forecasting project exploring multiple LSTM architectures for retail sales and stock price prediction. The project contains several standalone scripts, each implementing a different forecasting approach.

## Script Descriptions & Architecture

### `main.py` — Stock price forecasting
- Loads `MicrosoftStock.csv`, applies StandardScaler, uses a 60-day sliding window
- Sequential LSTM: 2×LSTM(64) → Dense(128, ReLU) → Dropout(0.5) → Dense(1)
- Plots actual vs predicted closing prices

### `test.py` — Single-series retail sales LSTM
- Reads `dataset/sales_train_validation.csv`, selects one item/store combo (index 6780)
- MinMaxScaler to [-1, 1], 28-day window, 67/33 split, same architecture as `main.py`

### `global_model_lstm.py` — Multi-series global LSTM (most advanced)
- Trains one model across up to `MAX_SERIES=500` series simultaneously
- Architecture adds a **series embedding layer** (16-dim): LSTM output is concatenated with the embedding before the Dense head
- Saves/loads artifacts from `artifacts_global_lstm/` (model, scalers, metadata) to avoid retraining
- To retrain from scratch, delete or set `LOAD_ARTIFACTS = False`

### `prediction.py` — Single-product forecasting with feature engineering
- CLI: `--product_id`, `--window`, `--epochs`, `--batch_size`, `--train_path`, `--test_path`
- Adds lag features (lag-1, lag-7, lag-14) and one-hot encodes Seasonality/Category
- Fits scalers **only on train data**; aligns feature columns between train and test
- Reports RMSE, MAE, R² and plots predictions + training loss curves

### `dataset_preparation.py` — Data splitting utility
- Reads `dataset/retail_store_inventory.csv`
- Performs a **temporal 80/20 split** by date (split date: 2023-08-08)
- Outputs `dataset/split/train.csv`, `dataset/split/test.csv`, and `dataset/split_config.json`

## Data Flow

```
retail_store_inventory.csv
        ↓  dataset_preparation.py
dataset/split/{train,test}.csv
        ↓  prediction.py (per product)
Lag features + one-hot encoding + MinMaxScaler
        ↓
LSTM → evaluate (RMSE, MAE, R²) → plot
```

The large dataset files (`dataset/`) are git-ignored and must be sourced separately.

## Key Configuration Constants

| Script | Constant | Default | Purpose |
|---|---|---|---|
| `global_model_lstm.py` | `MAX_SERIES` | 500 | Limit series for faster runs |
| `global_model_lstm.py` | `LOAD_ARTIFACTS` | True | Skip training if artifacts exist |
| `global_model_lstm.py` | `SEQ_LENGTH` | 28 | Sliding window size |
| `test.py` / `main.py` | `SEQ_LENGTH` | 28/60 | Sliding window size |

## Trained Artifacts

`artifacts_global_lstm/` contains a pre-trained global model:
- `model.keras` — trained Keras model
- `scaler.npz` — fitted per-series MinMaxScalers
- `meta.json` — series IDs and training configuration
