# Model Walkthrough

This document explains the final `aicup_solar_xgb.py` file step by step, including which parts come from the old project idea and which parts are new cleanup/improvement work.

## High-Level Idea

The final model keeps the old two-stage design:

```text
Step 1: timestamp + location -> weather features
Step 2: weather features + time/location features -> solar power
```

Why two stages?

The submission file only gives an id. The id contains date, time, and location, but it does not directly provide `Temperature`, `Humidity`, `Sunlight`, or `Pressure`. The old project idea was to first infer these hidden weather features, then use them to predict power. That idea is still the core of the current version.

## Old Thinking Kept

- Use XGBoost because the relationship is nonlinear.
- Use two stages instead of directly predicting power from date.
- In Step 1, predict hidden weather features.
- In Step 2, predict `Power`.
- Keep the best old Step 2 hyperparameters: `n_estimators=2000`, `learning_rate=0.01`, `max_depth=8`, `reg_lambda=15`, `reg_alpha=5`.

## New Thinking Added

- Preserve real `LocationCode` after resampling.
- Use a temporal validation split instead of only random split.
- Add cyclic time features: `TimeSin`, `TimeCos`, `YearSin`, `YearCos`.
- Let Step 2 also use time and location features, not only weather features.
- Support additional V2 training data.
- Compare legacy and cleaned models under the same validation split.

## Function-by-Function Explanation

### `read_training_csv`

Reads one training CSV and renames columns into a consistent schema:

```text
LocationCode, DateTime, WindSpeed, Pressure, Temperature, Humidity, Sunlight, Power
```

This is cleanup work. The old code also renamed columns, but this version wraps that logic into a reusable function.

### `load_training_data`

Loads every CSV in `data/train`, and optionally loads additional V2 data from `data/train_additional_v2`.

Old idea: read all training CSV files.

New addition: support the additional V2 training folder in the same flow.

### `resample_10min`

Converts minute-level data into 10-minute averages.

Old idea: resample by 10 minutes.

New fix: resample per `LocationCode`, then restore the correct location code. This avoids the old problem where all rows were accidentally treated as location 1.

### `add_time_features`

Adds basic time features:

```text
Year, Month, Day, Hour, Minute, DayOfYear
```

It also adds cyclic features:

```text
TimeSin, TimeCos, YearSin, YearCos
```

Basic time features are from the old idea. Cyclic features are new. They help the model understand that time is circular, such as 23:50 being close to 00:00.

### `parse_submission_ids`

Parses the submission id into:

```text
DateTime, LocationCode
```

The id format is:

```text
YYYYMMDDHHMMLL
```

where `LL` is the location code.

### `time_feature_columns`

Defines the cleaned Step 1 input columns.

Cleaned Step 1 uses normal time features, cyclic time features, and `LocationCode`.

### `legacy_time_feature_columns`

Defines the old Step 1 input columns:

```text
Year, Month, Day, Hour, Minute, LocationCode
```

This is kept only for fair baseline comparison.

### `power_feature_columns`

Defines the cleaned Step 2 input columns.

Cleaned Step 2 uses predicted weather plus time and location features. This is new. It helps because solar output depends not only on weather, but also on device/location differences and time-of-day or seasonal patterns.

### `legacy_power_feature_columns`

Defines the old Step 2 input columns:

```text
Temperature, Humidity, Sunlight, Pressure
```

This represents the old approach, where Step 2 only used weather features.

### `make_weather_model`

Creates the cleaned Step 1 XGBoost model.

It predicts hidden weather features from timestamp/location features.

### `make_legacy_weather_model`

Creates the old Step 1 XGBoost model from the experiment record. It is used only for comparison.

### `make_power_model`

Creates the cleaned Step 2 XGBoost model.

The hyperparameters are based on the best old recorded Step 2 experiment, but the feature set is improved.

### `make_legacy_power_model`

Creates the old Step 2 XGBoost model for baseline comparison.

### `train_weather_models`

Trains one Step 1 model for each hidden weather target:

```text
Temperature, Humidity, Sunlight, Pressure
```

### `predict_weather`

Uses the Step 1 models to predict hidden weather features for validation rows or submission rows.

It also clips impossible values:

- `Humidity` is clipped to 0-100.
- `Sunlight` is clipped to 0 or above.

### `train_power_model`

Trains Step 2:

```text
weather + time/location -> Power
```

The Step 2 idea is old, but adding time/location into Step 2 is new.

### `regression_metrics`

Calculates:

- `MAE`
- `MSE`
- `RMSE`
- `R2`

`MSE` and `R2` match the old experiment record. `MAE` and `RMSE` are easier to interpret.

### `temporal_split`

Splits data by time:

```text
older records -> training
newer records -> validation
```

This is new. It is stricter and more realistic than random split because the competition is closer to future forecasting.

### `evaluate_cleaned_two_stage`

Evaluates the final cleaned model:

1. Train Step 1 weather models on training data.
2. Predict weather features for validation rows.
3. Replace validation weather features with predicted weather features.
4. Train Step 2 power model.
5. Predict validation power.
6. Calculate metrics.

Replacing validation weather with predicted weather is important because the submission file does not provide real weather values.

### `evaluate_legacy_two_stage`

Evaluates the old two-stage approach using the same temporal validation rows.

It keeps old choices:

- old time features
- `StandardScaler`
- old Step 1 parameters
- Step 2 uses only weather features
- old best Step 2 parameters

This makes the old/new comparison fair.

### `evaluate`

Default evaluation entry point. It currently evaluates the cleaned two-stage model.

### `compare_baselines`

Runs both models on the same split and returns a comparison table.

Current comparison result:

| Model | MAE | MSE | RMSE | R2 | valid_rows |
|---|---:|---:|---:|---:|---:|
| legacy_two_stage_xgb | 223.9986 | 175087.0701 | 418.4341 | 0.4059 | 28094 |
| cleaned_two_stage_xgb | 181.5689 | 109734.2368 | 331.2616 | 0.6276 | 28094 |

### `predict_submission`

Trains on all available processed data and writes predictions into the submission template.

Steps:

1. Read submission CSV.
2. Parse ids into `DateTime` and `LocationCode`.
3. Add time features.
4. Predict hidden weather features.
5. Predict power.
6. Clip negative power predictions to 0.
7. Save output CSV.

## Final Result Interpretation

The fair comparison showed:

```text
MAE: 223.9986 -> 181.5689
MSE: 175087.0701 -> 109734.2368
RMSE: 418.4341 -> 331.2616
R2: 0.4059 -> 0.6276
```

This means the cleaned model is not only cleaner code. It is also more accurate under the same validation condition.

The strongest explanation is:

- location information was preserved correctly
- Step 2 learned from location and time, not only weather
- cyclic time features gave the model better daily/yearly structure
- temporal validation made the comparison more realistic
