# Experiment Summary

## Ver1

Direct prediction from date-derived features to power using `LinearRegression`.

- MSE: `219724.41`
- R2: `0.01`

Result: too weak for the nonlinear relation between sunlight/weather/time and generated power.

## Ver2

Direct prediction from date-derived features to power using `XGBRegressor`.

Best recorded configuration:

```python
XGBRegressor(
    n_estimators=200,
    learning_rate=0.3,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
)
```

- MSE: `86583.47`
- R2: `0.61`

## Ver3

Two-stage prediction:

1. Date/time/location to hidden weather features.
2. Weather features to generated power.

Best recorded power model:

```python
XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=15,
    reg_alpha=5,
    random_state=0,
)
```

- Step 1 hidden feature R2: about `0.64`
- Step 2 Power MSE: `85516.88`
- Step 2 Power R2: `0.62`

## Cleanup Changes

- Preserved the real `LocationCode` after resampling.
- Replaced hard-coded Colab paths with editable Colab-friendly settings.
- Added temporal validation instead of random split.
- Added cyclic time features for daily and annual seasonality.
- Added optional support for additional V2 training data.
- Kept the implementation in one Python file for easier review and reproduction.
