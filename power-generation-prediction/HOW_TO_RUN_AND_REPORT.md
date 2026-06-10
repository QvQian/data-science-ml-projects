# How to Test, Evaluate, and Report

This guide explains how to run the model in Colab, evaluate it, record results, and write a short report.

## 1. Prepare Data

Use this local project structure:

```text
data/
|-- train/
|   |-- L1_Train.csv
|   |-- L2_Train.csv
|   `-- ...
|-- train_additional_v2/
|   |-- L2_Train_2.csv
|   `-- ...
`-- submission_template/
    `-- upload(no answer).csv
```

The CSV files are ignored by Git because they are competition data.

## 2. Install Packages

In Colab:

```python
!pip install -r requirements.txt
```

## 3. Run Validation

The code is written for Colab UI Run. Open `aicup_solar_xgb.py`, edit the settings at the top, and run the whole block/file.

To compare the old and cleaned models:

```python
SKIP_VALIDATION = False
COMPARE_BASELINES = True
```

This prints a table like:

```text
                model      mae         mse     rmse     r2  valid_rows
 legacy_two_stage_xgb 223.9986 175087.0701 418.4341 0.4059       28094
cleaned_two_stage_xgb 181.5689 109734.2368 331.2616 0.6276       28094
```

To validate only the cleaned model:

```python
SKIP_VALIDATION = False
COMPARE_BASELINES = False
```

This prints metrics such as:

```json
{
  "mae": 181.56,
  "mse": 109734.23,
  "rmse": 331.26,
  "r2": 0.6276,
  "valid_rows": 28094
}
```

Record the final numbers in `RESULTS.md`.

## 4. Generate Final Submission

After validation, train on all available data and generate the output CSV:

```python
SKIP_VALIDATION = True
COMPARE_BASELINES = False
```

The generated file is saved to:

```text
outputs/submission_xgb.csv
```

## 5. What to Evaluate

Use these metrics:

- `MAE`: average absolute prediction error
- `MSE`: squared error, useful for comparing with old experiment records
- `RMSE`: square root of MSE, easier to interpret because it has the same unit as power
- `R2`: how much variance the model explains

For this project, the most important comparison is:

```text
legacy_two_stage_xgb vs cleaned_two_stage_xgb
```

Both should be evaluated on the same temporal validation split.

## 6. How to Write Results

Use a table:

| Model | Validation Strategy | Extra Data | MAE | MSE | RMSE | R2 | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Legacy two-stage XGB | temporal split | Yes | 223.9986 | 175087.0701 | 418.4341 | 0.4059 | Old-style pipeline rerun under same split |
| Cleaned two-stage XGB | temporal split | Yes | 181.5689 | 109734.2368 | 331.2616 | 0.6276 | Final cleaned model |

## 7. Suggested Report Structure

### Motivation

This project predicts solar power generation from timestamp, location, and weather sensor data. Solar generation is nonlinear because it depends strongly on sunlight, time of day, season, humidity, and local device conditions.

### Data Processing

Raw minute-level readings from 17 locations were resampled into 10-minute intervals to match the submission format. Additional V2 training data was optionally included to improve coverage for later test dates.

### Method

The final model uses a two-stage XGBoost pipeline. The first stage predicts hidden weather features from timestamp and location. The second stage predicts solar power using predicted weather features plus time and location features.

### Validation

A temporal validation split was used to better simulate future prediction. Older records were used for training, and later records were used for validation.

### Results

Under the same temporal split, the cleaned model reduced RMSE from `418.43` to `331.26` and improved R2 from `0.4059` to `0.6276`.

### Discussion

The cleaned model improves the old two-stage idea by preserving real location information and adding cyclic time features. These features help the model learn daily and seasonal solar generation patterns.

### Limitations

The first-stage weather prediction can introduce error into the second stage. Validation results also depend on how the time split is chosen.

### Future Work

Possible improvements include per-location models, LightGBM/CatBoost comparison, solar-angle features, and model ensembling.
