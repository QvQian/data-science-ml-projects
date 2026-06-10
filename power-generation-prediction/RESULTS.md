# Results

## Environment

- Platform: Google Colab
- Resampling: 10-minute average
- Validation: temporal split
- Additional data: included

## Fair Comparison

Both models were rerun on the same temporal validation split.

| Model | Validation Strategy | Extra Data | MAE | MSE | RMSE | R2 | Validation Rows | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Legacy two-stage XGB | temporal split | Yes | 223.9986 | 175087.0701 | 418.4341 | 0.4059 | 28094 | Old-style two-stage pipeline |
| Cleaned two-stage XGB | temporal split | Yes | 181.5689 | 109734.2368 | 331.2616 | 0.6276 | 28094 | Final cleaned model |

## Interpretation

The cleaned model performed better under the same validation condition:

- MAE decreased from `223.9986` to `181.5689`.
- MSE decreased from `175087.0701` to `109734.2368`.
- RMSE decreased from `418.4341` to `331.2616`.
- R2 improved from `0.4059` to `0.6276`.

This suggests that preserving real location information and adding cyclic time/location features improved generalization to later records.

## Historical Experiment Records

Older experiment records used different validation settings, so they are useful as development history but should not be directly compared against the fair temporal-split table above.

| Model | Validation Strategy | MSE | R2 | Notes |
|---|---|---:|---:|---|
| Linear Regression | old random split | 219724.41 | 0.01 | Initial baseline |
| Direct XGBoost | old random split | 86583.47 | 0.61 | Nonlinear model improved strongly |
| Two-stage XGBoost | old random split | 85516.88 | 0.62 | Best recorded old experiment |

## Final Submission

- Output file: `outputs/submission_xgb.csv`
- Leaderboard score: not recorded
