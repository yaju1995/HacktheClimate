# HacktheClimate wind-power forecasting

This project loads time-series CSV data, validates it, creates lookback/horizon
sequences, and trains an LSTM model to forecast wind generation. The primary
walkthrough is the notebook [`test_dataload_syn_data.ipynb`](./test_dataload_syn_data.ipynb),
which uses the synthetic Ireland national hourly dataset.

## 1. Get the project

Pull the branch you want from the remote. Replace `main` with the branch name
you need:

```bash
git pull origin main
```

For a first checkout, clone the repository and then enter its directory:

```bash
git clone https://github.com/yaju1995/HacktheClimate.git
cd HacktheClimate
git pull origin main
```

Run all commands below from the repository root (the directory containing
`src`, `Data`, and the notebook).

## 2. Create and activate `.venv`

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks script activation, either allow scripts for the current
PowerShell session or use the activation batch file:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

```bat
.venv\Scripts\activate.bat
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install requirements

With `.venv` activated, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The requirements cover pandas/NumPy data handling, scikit-learn scaling and
metrics, PyTorch LSTM training, matplotlib plots, and Jupyter notebook
execution.

## 4. Store and select the data

Keep input CSV files under the repository's `Data/` directory. The notebook
currently selects:

```python
csv_path = "Data/synthetic_ireland_national_hourly.csv"
```

To use another file, copy it into `Data/` and change `csv_path` in the
`LoadForecastDataLoader` cell. For example:

```python
csv_path = "Data/my_hourly_wind_data.csv"
```

The path is relative to the repository root, so start Jupyter from that
directory. The loader accepts only files with a `.csv` extension.

## 5. Required CSV format

The CSV must:

- include one header row;
- have unique, non-empty column names (leading/trailing header whitespace is
  removed);
- contain one row per time step;
- contain one timestamp column, the configured feature columns, and the target
  column;
- contain numeric values in every configured feature and target column;
- contain no missing values in the timestamp, feature, or target columns;
- use a regular time interval matching `expected_frequency` (the notebook uses
  `1h`);
- have no duplicate timestamps.

The notebook's default configuration expects these columns:

```text
timestamp_local
hour
day_of_week
month
mean_wind_speed
wind_capacity_weighted_wind_speed
wind_speed_squared
wind_speed_lag_1h
mean_temperature
temperature_lag_1h
wind_generation_mw
```

The CSV may contain additional columns; only the configured columns are used.
The default feature list and target are:

```python
features = [
    "hour",
    "day_of_week",
    "month",
    "mean_wind_speed",
    "wind_capacity_weighted_wind_speed",
    "wind_speed_squared",
    "wind_speed_lag_1h",
    "mean_temperature",
    "temperature_lag_1h",
]
target = "wind_generation_mw"
```

All configured feature and target values must be numeric. Do not write units
such as `MW` or `m/s` into numeric cells; keep units in the column names or
documentation.

### Timestamp format

The synthetic notebook data uses timezone-aware ISO 8601 timestamps, for
example:

```text
2022-01-01T00:00:00+00:00
2022-01-01T01:00:00+00:00
```

Because the notebook leaves `datetime_format=None`, pandas infers this standard
format. Timestamps must be parseable and should be in chronological order.
The loader sorts them when `sort_by_datetime=True`, but duplicate timestamps
and gaps are still reported as data abnormalities.

For a non-standard timestamp format, set `datetime_format` explicitly. For
example, `T1.csv` uses values such as `01 01 2018 00:00`, so its configuration
uses:

```python
datetime_col = "Date/Time"
datetime_format = "%d %m %Y %H:%M"
```

The format codes mean: day (`%d`), month (`%m`), year (`%Y`), hour (`%H`), and
minute (`%M`). The `T1.csv` file is sampled every 10 minutes and therefore
does not match the notebook's `expected_frequency="1h"` without resampling.

## 6. Open and run the notebook

From the repository root with `.venv` activated:

```bash
python -m jupyter notebook
```

Open [`test_dataload_syn_data.ipynb`](./test_dataload_syn_data.ipynb), select
the `.venv` Python kernel, and run the cells from top to bottom. In VS Code,
open the notebook directly, choose the `.venv` interpreter/kernel when
prompted, and use **Run All**.

The notebook will:

1. create the model output directory;
2. load and validate the selected CSV;
3. scale the features and target;
4. create LSTM windows using `lookback=48` and `horizon=24`;
5. train and evaluate the model;
6. save the best model checkpoint.

If the loader reports missing columns, invalid timestamps, non-numeric values,
missing values, duplicate timestamps, or time gaps, fix the CSV or update the
loader configuration before training.

## 7. Available models

### LSTM

[`src/models/lstm.py`](./src/models/lstm.py) provides `LSTMForecast` for
sequence-based forecasting. It accepts inputs shaped
`[batch, lookback, features]` and predicts a configurable forecast horizon.
`LSTMDataset` creates the lookback and horizon windows used by the LSTM
workflow.

### Feed-forward model

[`src/models/feedforward.py`](./src/models/feedforward.py) provides
`FeedForwardForecast` for instant-to-instant prediction. It accepts one row of
features at a time, shaped `[batch, input_size]`, and returns
`[batch, horizon, 1]`. For the wind-generation example, the model uses the
nine configured instant features and `horizon=1` to predict the generation at
the same timestamp. It does not create or consume an LSTM lookback window.

### Transformer

[`src/models/transformer.py`](./src/models/transformer.py) provides
`TransformerForecast` using a Transformer encoder. It projects input features
into the Transformer dimension, processes the sequence, uses the final
timestep representation, and maps it to the forecast horizon. Its expected
input is a sequence shaped `[batch, sequence_length, features]`, and its
output is `[batch, horizon, 1]`.

## 8. Feed-forward training notebook

[`train_feedforward.ipynb`](./train_feedforward.ipynb) trains and tests the
feed-forward model with the synthetic Ireland hourly dataset. The notebook:

1. loads and validates data through `LoadForecastDataLoader`;
2. keeps data from 2022 through 2024;
3. separates the data chronologically into 70% train, 15% validation, and 15%
   test sets;
4. fits feature and target scalers on the training set only;
5. trains `FeedForwardForecast` using instant features to predict instant
   `wind_generation_mw`;
6. saves the best checkpoint to
   `models/generation_feedforward/best_feedforward.pt`;
7. evaluates the test set with MAE and RMSE; and
8. visualizes the final 24 hours of the test set with actual and predicted
   generation values.

Open the notebook from the repository root, select the `.venv` Python kernel,
and run the cells from top to bottom. Update the CSV path in the loader cell
when using a different compatible dataset.

## 9. Model output

The notebook creates this directory automatically:

```text
models/24_lookback_24_horizon_syn/
```

The best model is saved as:

```text
models/24_lookback_24_horizon_syn/best_lstm.pt
```

The notebook's save cell uses:

```python
SAVE_DIR = Path("models/24_lookback_24_horizon_syn")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = SAVE_DIR / "best_lstm.pt"
```

Keep model files under `models/` (or another explicitly configured output
directory), and do not save them inside `Data/`. When changing the lookback,
horizon, feature list, or dataset, use a separate output directory so
checkpoints are not accidentally overwritten. The saved `.pt` file is a
PyTorch state dictionary and must be loaded with the same model architecture
and compatible feature/horizon configuration.

## Optional command-line training

The standalone script uses the separate `Data/T1.csv` schema by default:

```bash
python train_lstm.py --mode train --data Data/T1.csv --save-dir artifacts
```

To evaluate the saved checkpoint:

```bash
python train_lstm.py --mode evaluate --data Data/T1.csv --save-dir artifacts
```

The command-line script expects `Date/Time`, `LV ActivePower (kW)`, and its
three configured feature columns. Use the notebook configuration instead when
working with `synthetic_ireland_national_hourly.csv`.
