# AICUP 2024 Solar Power Forecasting

Cleaned and reproducible two-stage XGBoost project for solar power generation prediction.

The original project was developed for the 2024 AICUP solar power forecasting competition. This repository keeps the original two-stage modeling idea, then cleans up the data pipeline, fixes location handling, adds time features, and evaluates the old and cleaned models under the same temporal validation split.

## Method

The model uses a two-stage pipeline:

```text
Step 1: timestamp + location -> weather features
Step 2: weather features + time/location features -> solar power
```

The submission file only provides an id containing date, time, and location. It does not provide direct weather measurements, so Step 1 predicts hidden weather features first:

- `Temperature`
- `Humidity`
- `Sunlight`
- `Pressure`

Step 2 predicts `Power` from predicted weather features plus time and location features.

## Legacy vs Cleaned Model

| Component | Legacy two-stage XGB | Cleaned two-stage XGB |
|---|---|---|
| Step 1 input | Basic time + location | Basic time + cyclic time + location |
| Step 1 target | Weather features | Weather features |
| Step 2 input | Weather features only | Weather + time + cyclic time + location |
| Location handling | Original code could overwrite location as 1 | Preserves real `LocationCode` |
| Validation | Old records used random split | Uses temporal split |

## Results

Both models were rerun on the same temporal validation split.

| Model | MAE | MSE | RMSE | R2 | Validation Rows |
|---|---:|---:|---:|---:|---:|
| Legacy two-stage XGB | 223.9986 | 175087.0701 | 418.4341 | 0.4059 | 28094 |
| Cleaned two-stage XGB | 181.5689 | 109734.2368 | 331.2616 | 0.6276 | 28094 |

The cleaned model reduced RMSE from `418.43` to `331.26` and improved R2 from `0.4059` to `0.6276` under the same validation setting.

## Repository Layout

```text
power-generation-prediction/
|-- aicup_solar_xgb.py
|-- README.md
|-- RESULTS.md
|-- MODEL_WALKTHROUGH.md
|-- HOW_TO_RUN_AND_REPORT.md
|-- experiments.md
|-- requirements.txt
|-- data/
|   |-- README.md
|   |-- train/
|   |-- train_additional_v2/
|   `-- submission_template/
`-- outputs/
    `-- README.md
```

Raw competition CSV files and generated prediction outputs are not committed.

## How to Run in Colab

1. Put the project folder in Google Drive.
2. In Colab, change directory to the project folder:

```python
%cd /content/drive/MyDrive/AICUP_2026ver
```

3. Install dependencies:

```python
!pip install -r requirements.txt
```

4. Open `aicup_solar_xgb.py`, check the settings at the top, then run the whole block/file:

```python
TRAIN_DIR = Path("data/train")
ADDITIONAL_DIR = Path("data/train_additional_v2")
UPLOAD_FILE = Path("data/submission_template/upload(no answer).csv")
OUTPUT_FILE = Path("outputs/submission_xgb.csv")

SKIP_VALIDATION = False
COMPARE_BASELINES = True
```

Use these modes:

```python
# Compare old and cleaned models
SKIP_VALIDATION = False
COMPARE_BASELINES = True
```

```python
# Validate only the cleaned model
SKIP_VALIDATION = False
COMPARE_BASELINES = False
```

```python
# Generate final submission with all data
SKIP_VALIDATION = True
COMPARE_BASELINES = False
```

## More Documentation

- `RESULTS.md`: validation results and experiment notes
- `MODEL_WALKTHROUGH.md`: step-by-step explanation of every function
- `HOW_TO_RUN_AND_REPORT.md`: testing, evaluation, and report-writing guide
- `experiments.md`: summary of older experiment records
