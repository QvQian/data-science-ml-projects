import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


COLUMNS = [
    "LocationCode",
    "DateTime",
    "WindSpeed",
    "Pressure",
    "Temperature",
    "Humidity",
    "Sunlight",
    "Power",
]

WEATHER_TARGETS = ["Temperature", "Humidity", "Sunlight", "Pressure"]


# Colab-friendly settings
TRAIN_DIR = Path("data/train")
ADDITIONAL_DIR = Path("data/train_additional_v2")
UPLOAD_FILE = Path("data/submission_template/upload(no answer).csv")
OUTPUT_FILE = Path("outputs/submission_xgb.csv")

SKIP_VALIDATION = False # Set True only when training on all data and produce final submission
COMPARE_BASELINES = True # True = print old-vs-cleaned comparison table using the same temporal split


def read_training_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = COLUMNS
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"])
    return df


# Load original training CSVs + optional additional V2 data
def load_training_data(train_dir: Path, additional_dir: Path | None = None) -> pd.DataFrame:
    files = sorted(train_dir.glob("*.csv"))
    if additional_dir and additional_dir.exists():
        files += sorted(additional_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No training CSV files were found.")

    frames = [read_training_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def resample_10min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute-level sensor rows into 10-minute averages.

    New fix: resample per location and keep the real LocationCode. The old
    version accidentally set all rows to LocationCode=1 after grouping.
    """
    grouped = []
    for location, part in df.groupby("LocationCode", sort=True):
        part = part.sort_values("DateTime").set_index("DateTime")
        ten_min = part.resample("10min").mean(numeric_only=True)
        ten_min["LocationCode"] = int(location)
        grouped.append(ten_min.reset_index())

    out = pd.concat(grouped, ignore_index=True)
    out = out.round(2).dropna()
    return out.sort_values(["LocationCode", "DateTime"]).reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["DateTime"]
    out["Year"] = dt.dt.year
    out["Month"] = dt.dt.month
    out["Day"] = dt.dt.day
    out["Hour"] = dt.dt.hour
    out["Minute"] = dt.dt.minute
    out["DayOfYear"] = dt.dt.dayofyear

    # Cyclic features express daily/yearly loops better than plain integers.
    minutes = out["Hour"] * 60 + out["Minute"]
    out["TimeSin"] = np.sin(2 * np.pi * minutes / 1440)
    out["TimeCos"] = np.cos(2 * np.pi * minutes / 1440)
    out["YearSin"] = np.sin(2 * np.pi * out["DayOfYear"] / 366)
    out["YearCos"] = np.cos(2 * np.pi * out["DayOfYear"] / 366)
    return out


def parse_submission_ids(upload: pd.DataFrame) -> pd.DataFrame:
    id_column = "序號" if "序號" in upload.columns else upload.columns[0]

    ids = upload[id_column].astype(str).str.zfill(14)
    parsed = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(ids.str[:12], format="%Y%m%d%H%M", errors="coerce"),
            "LocationCode": ids.str[12:14].astype(int),
        }
    )
    return parsed

# Cleaned Step 1 inputs: time, cyclic time, and location
def time_feature_columns() -> list[str]:
    return [
        "Year",
        "Month",
        "Day",
        "Hour",
        "Minute",
        "DayOfYear",
        "TimeSin",
        "TimeCos",
        "YearSin",
        "YearCos",
        "LocationCode",
    ]

# Old Step 1 inputs kept for fair baseline comparison
def legacy_time_feature_columns() -> list[str]:
    return [
        "Year",
        "Month",
        "Day",
        "Hour",
        "Minute",
        "LocationCode",
    ]

# Cleaned Step 2 inputs: weather plus time/location features.
def power_feature_columns() -> list[str]:
    return WEATHER_TARGETS + [
        "Month",
        "Hour",
        "Minute",
        "DayOfYear",
        "TimeSin",
        "TimeCos",
        "YearSin",
        "YearCos",
        "LocationCode",
    ]

# Old Step 2 inputs: weather features only
def legacy_power_feature_columns() -> list[str]:
    return WEATHER_TARGETS

# Create the cleaned Step 1 model for weather prediction
def make_weather_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=-1,
    )

# Create a legacy-style Step 1 model based on the experiment record
def make_legacy_weather_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=200,
        learning_rate=0.3,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=-1,
    )

# Create the cleaned Step 2 model for power prediction
def make_power_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=15,
        reg_alpha=5,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=-1,
    )

# Create the old best-recorded Step 2 model
def make_legacy_power_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=15,
        reg_alpha=5,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=-1,
    )

# Train one Step 1 XGBoost model for each hidden weather target
def train_weather_models(train_df: pd.DataFrame) -> dict[str, XGBRegressor]:
    x = train_df[time_feature_columns()]
    models = {}
    for target in WEATHER_TARGETS:
        model = make_weather_model()
        model.fit(x, train_df[target])
        models[target] = model
    return models

# Predict hidden weather features from timestamp/location features
def predict_weather(models: dict[str, XGBRegressor], feature_df: pd.DataFrame) -> pd.DataFrame:
    x = feature_df[time_feature_columns()]
    pred = pd.DataFrame(index=feature_df.index)
    for target, model in models.items():
        pred[target] = model.predict(x)

    pred["Humidity"] = pred["Humidity"].clip(0, 100)
    pred["Sunlight"] = pred["Sunlight"].clip(lower=0)
    return pred

# Train Step 2: weather plus time/location features -> Power
def train_power_model(train_df: pd.DataFrame) -> XGBRegressor:
    model = make_power_model()
    model.fit(train_df[power_feature_columns()], train_df["Power"])
    return model

# Calculate validation metrics used in reports
def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }

# Use older rows for training and newer rows for validation
def temporal_split(df: pd.DataFrame, valid_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["DateTime"].quantile(1 - valid_ratio)
    train_df = df[df["DateTime"] <= cutoff].copy()
    valid_df = df[df["DateTime"] > cutoff].copy()
    return train_df, valid_df


def evaluate_cleaned_two_stage(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict[str, float]:
    weather_models = train_weather_models(train_df)
    weather_pred = predict_weather(weather_models, valid_df)

    # Validation should mimic submission: Step 2 receives predicted weather,
    # not the real validation weather measurements.
    valid_power_input = valid_df.drop(columns=WEATHER_TARGETS).join(weather_pred)
    power_model = train_power_model(train_df)
    pred_power = power_model.predict(valid_power_input[power_feature_columns()])
    pred_power = np.maximum(pred_power, 0)

    metrics = regression_metrics(valid_df["Power"], pred_power)
    metrics["valid_rows"] = int(len(valid_df))
    return metrics


def evaluate_legacy_two_stage(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict[str, float]:
    x_weather_train = train_df[legacy_time_feature_columns()]
    x_weather_valid = valid_df[legacy_time_feature_columns()]
    y_weather_train = train_df[WEATHER_TARGETS]

    weather_scaler = StandardScaler()
    x_weather_train_scaled = weather_scaler.fit_transform(x_weather_train)
    x_weather_valid_scaled = weather_scaler.transform(x_weather_valid)

    weather_model = make_legacy_weather_model()
    weather_model.fit(x_weather_train_scaled, y_weather_train)
    pred_weather = weather_model.predict(x_weather_valid_scaled)

    power_scaler = StandardScaler()
    x_power_train_scaled = power_scaler.fit_transform(train_df[legacy_power_feature_columns()])
    pred_weather_scaled = power_scaler.transform(pred_weather)

    power_model = make_legacy_power_model()
    power_model.fit(x_power_train_scaled, train_df["Power"])
    pred_power = power_model.predict(pred_weather_scaled)
    pred_power = np.maximum(pred_power, 0)

    metrics = regression_metrics(valid_df["Power"], pred_power)
    metrics["valid_rows"] = int(len(valid_df))
    return metrics


def evaluate(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict[str, float]:
    return evaluate_cleaned_two_stage(train_df, valid_df)


def compare_baselines(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"model": "legacy_two_stage_xgb", **evaluate_legacy_two_stage(train_df, valid_df)},
        {"model": "cleaned_two_stage_xgb", **evaluate_cleaned_two_stage(train_df, valid_df)},
    ]
    return pd.DataFrame(rows)


def predict_submission(train_df: pd.DataFrame, upload_path: Path, output_path: Path) -> None:
    upload = pd.read_csv(upload_path)
    parsed = parse_submission_ids(upload)
    parsed = add_time_features(parsed)

    weather_models = train_weather_models(train_df)
    weather_pred = predict_weather(weather_models, parsed)
    power_input = parsed.join(weather_pred)

    power_model = train_power_model(train_df)
    pred = power_model.predict(power_input[power_feature_columns()])
    upload["答案"] = np.round(np.maximum(pred, 0), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    upload.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.2f")



raw = load_training_data(TRAIN_DIR, ADDITIONAL_DIR)
train_all = add_time_features(resample_10min(raw))

if not SKIP_VALIDATION:
    train_df, valid_df = temporal_split(train_all)
    if COMPARE_BASELINES:
        comparison = compare_baselines(train_df, valid_df)
        print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    else:
        metrics = evaluate(train_df, valid_df)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

predict_submission(train_all, UPLOAD_FILE, OUTPUT_FILE)
print(f"Saved submission to {OUTPUT_FILE}")
