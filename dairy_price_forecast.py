import argparse
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor


def parse_args():
    p = argparse.ArgumentParser(description="Dairy product price prediction")
    p.add_argument("--csv", required=True, help="Path to dataset CSV")
    p.add_argument("--date-col", required=True, help="Name of date column")
    p.add_argument("--target-col", required=True, help="Name of price/target column")

    # Optional categorical columns present in your CSV
    p.add_argument("--product-col", default=None, help="Name of product column (optional)")
    p.add_argument("--region-col", default=None, help="Name of region column (optional)")

    # Optional filters to build a single series
    p.add_argument("--product", default=None, help="Filter to this product value (optional)")
    p.add_argument("--region", default=None, help="Filter to this region value (optional)")

    p.add_argument("--test-size", type=float, default=0.2,
                   help="Fraction of tail data for final test (default 0.2)")
    p.add_argument("--n_splits", type=int, default=3, help="TimeSeriesSplit folds (default 3)")
    p.add_argument("--forecast-horizon", type=int, default=0,
                   help="Steps to forecast into the future (default 0 = no forecast)")
    p.add_argument("--date-freq", default=None,
                   help="Pandas frequency alias for future dates if needed (e.g., 'MS','D','W'). "
                        "If omitted, inferred from date index.")
    return p.parse_args()


def load_and_prepare(args):
    df = pd.read_csv(args.csv)
    # Basic cleaning
    if args.date_col not in df.columns:
        raise ValueError(f"'{args.date_col}' not in columns: {df.columns.tolist()}")
    if args.target_col not in df.columns:
        raise ValueError(f"'{args.target_col}' not in columns: {df.columns.tolist()}")

    # Filter to single product/region if requested
    if args.product_col and args.product:
        df = df[df[args.product_col].astype(str) == str(args.product)]
    if args.region_col and args.region:
        df = df[df[args.region_col].astype(str) == str(args.region)]
    if df.empty:
        raise ValueError("No rows remain after filtering. Check product/region values.")

    # Parse dates
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df = df.dropna(subset=[args.date_col, args.target_col])
    df = df.sort_values(args.date_col).reset_index(drop=True)

    # Aggregate duplicates on same date (if any)
    group_cols = [args.date_col]
    df = df.groupby(group_cols, as_index=False)[args.target_col].mean()

    df = df.set_index(args.date_col)

    # Try to infer frequency (helps forecasting)
    try:
        inferred = pd.infer_freq(df.index)
    except Exception:
        inferred = None

    return df, inferred


def add_time_features(s):
    """
    Build a feature DataFrame from a price Series s (indexed by DatetimeIndex).
    Includes calendar features + lags/rolling stats.
    """
    X = pd.DataFrame(index=s.index)
    X["year"] = X.index.year
    X["month"] = X.index.month
    X["quarter"] = X.index.quarter
    X["dayofweek"] = X.index.dayofweek

    # Lags (t-1, t-7, t-30)
    X["lag_1"] = s.shift(1)
    X["lag_7"] = s.shift(7)
    X["lag_30"] = s.shift(30)

    # Rolling means/stds
    X["roll_mean_7"] = s.shift(1).rolling(window=7, min_periods=3).mean()
    X["roll_std_7"] = s.shift(1).rolling(window=7, min_periods=3).std()
    X["roll_mean_30"] = s.shift(1).rolling(window=30, min_periods=10).mean()
    X["roll_std_30"] = s.shift(1).rolling(window=30, min_periods=10).std()

    return X


def train_validate(X, y, args):
    # Drop rows with NaNs introduced by lags/rollings
    valid_mask = ~X.isna().any(axis=1)
    X, y = X.loc[valid_mask], y.loc[valid_mask]

    # Train/Test split by time
    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Preprocess: all features numeric; keep a placeholder for extensibility
    pre = ColumnTransformer(
        transformers=[("passthrough", "passthrough", X.columns.tolist())],
        remainder="drop",
    )

    models = {
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42
        ),
        "HistGB": HistGradientBoostingRegressor(
            max_depth=None, learning_rate=0.05, max_iter=500, random_state=42
        ),
    }

    results = {}
    best_name, best_pipe, best_cv = None, None, np.inf
    tscv = TimeSeriesSplit(n_splits=args.n_splits)

    for name, model in models.items():
        pipe = make_pipeline(pre, model)
        # Use negative MAPE for CV scoring (lower is better)
        # MAPE is undefined when y=0; consider small epsilon
        eps = 1e-8
        scorer = "neg_mean_absolute_percentage_error"
        cv_scores = cross_val_score(pipe, X_train, np.maximum(y_train, eps),
                                    cv=tscv, scoring=scorer, n_jobs=-1)
        mean_mape = -cv_scores.mean()
        results[name] = mean_mape
        if mean_mape < best_cv:
            best_cv = mean_mape
            best_name = name
            best_pipe = pipe

    # Fit best model on all train data
    best_pipe.fit(X_train, y_train)

    # Evaluate on hold-out
    y_pred = best_pipe.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(np.maximum(y_test, 1e-8), np.maximum(y_pred, 1e-8))

    info = {
        "cv_mape_by_model": results,
        "best_model": best_name,
        "test_rmse": rmse,
        "test_mape": mape,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "y_pred": pd.Series(y_pred, index=X_test.index),
        "pipeline": best_pipe
    }
    return info


def plot_results(y, y_pred, title="Actual vs Predicted"):
    plt.figure(figsize=(11, 5))
    y.plot(label="Actual")
    y_pred.plot(label="Predicted")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def recursive_forecast(last_series, model_pipeline, horizon, freq):
    """
    last_series: pandas Series of historical target (indexed by DatetimeIndex)
    model_pipeline: trained pipeline
    horizon: steps to forecast
    freq: pandas frequency string (e.g., 'MS','D','W')
    """
    history = last_series.copy()
    future_index = pd.date_range(
        start=history.index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=horizon, freq=freq
    )

    forecasts = []

    for step, dt in enumerate(future_index, start=1):
        # Build features for ALL current history and take the last row for 'dt'
        X_all = add_time_features(history)
        X_step = X_all.loc[[history.index[-1]]]  # we compute features at the last known point
        # BUT we want features corresponding to the new date 'dt'.
        # Shift index to 'dt' for calendar features:
        X_step = X_step.copy()
        X_step.index = pd.DatetimeIndex([dt])
        X_step["year"] = dt.year
        X_step["month"] = dt.month
        X_step["quarter"] = pd.Timestamp(dt).quarter
        X_step["dayofweek"] = dt.weekday()

        # Lags & rollings must come from history (already aligned in X_step from add_time_features(history))
        # Predict
        y_hat = model_pipeline.predict(X_step)[0]
        forecasts.append(y_hat)

        # Append the prediction to history to update lags/rollings for next step
        history = pd.concat([history, pd.Series([y_hat], index=[dt])])

    return pd.Series(forecasts, index=future_index, name="forecast")


def main():
    args = parse_args()

    df, inferred_freq = load_and_prepare(args)
    target = df[args.target_col].astype(float)

    # Build features from the target itself (univariate + calendar)
    X = add_time_features(target)

    info = train_validate(X, target, args)

    print("\n=== Cross-validated MAPE (lower is better) ===")
    for k, v in info["cv_mape_by_model"].items():
        print(f"{k:>14s}: {v:.4f}")
    print(f"\nBest model: {info['best_model']}")
    print(f"Hold-out RMSE: {info['test_rmse']:.4f}")
    print(f"Hold-out MAPE: {info['test_mape']:.4f}")

    # Plot hold-out segment
    plot_results(
        y=pd.concat([info["y_train"], info["y_test"]]),
        y_pred=pd.concat([pd.Series(index=info["y_train"].index, dtype=float), info["y_pred"]]),
        title=f"Actual vs Predicted ({info['best_model']})"
    )

    # Save pipeline
    joblib.dump(info["pipeline"], "best_dairy_price_model.joblib")
    print("\nSaved trained pipeline to: best_dairy_price_model.joblib")

    # Forecast future if requested
    if args.forecast_horizon > 0:
        freq = args.date_freq or inferred_freq
        if freq is None:
            raise ValueError(
                "Could not infer date frequency. Please pass --date-freq (e.g., 'MS','D','W')."
            )
        full_series = df[args.target_col].astype(float)
        fc = recursive_forecast(
            last_series=full_series,
            model_pipeline=info["pipeline"],
            horizon=args.forecast_horizon,
            freq=freq
        )
        print(f"\n=== {args.forecast_horizon}-step Forecast ===")
        print(fc)

        # Plot forecast
        plt.figure(figsize=(11, 5))
        full_series.tail(200).plot(label="History")
        fc.plot(label="Forecast")
        plt.title(f"{args.forecast_horizon}-step Forecast")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Save forecast to CSV
        out_path = "forecast_output.csv"
        fc.to_csv(out_path, header=True)
        print(f"\nSaved forecast to: {out_path}")


if __name__ == "__main__":
    main()
