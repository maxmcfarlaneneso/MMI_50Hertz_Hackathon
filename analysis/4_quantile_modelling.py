# =============================================================================
# WORKSTREAM 3: QUANTILE MODELLING
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from data_wrangling import read_feature_set, create_day_ahead_forecast_split
import xgboost as xgb

DEFAULT_MODEL_PARAMS = {
    "linear": {"alpha": 0.1, "solver": "highs"},  # no regularisation
    "xgb": {"n_estimators": 200, "learning_rate": 0.1, "random_state": 42},
    "qrf": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 20,
        "n_jobs": -1,
        "random_state": 42,
    },
    "gam": {},
}


def fit_quantile_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantile: float,
    model_type: str,
    params=None
):
    """
    Fit a quantile forecasting model.

    model_type:
    - xgb
    - qrf
    - linear
    - gam

    Default model params based on model type.

    Returns
    -------
    fitted model
    """

    model_type = model_type.lower()
    if params is None:
        params = {**DEFAULT_MODEL_PARAMS[model_type]}

    # -------------------------------------------------------------------------
    # Linear Quantile Regression
    # -------------------------------------------------------------------------

    if model_type == "linear":

        model = QuantileRegressor(
            quantile=quantile,
            **params,
        )
        model.fit(X_train, y_train)

        return model

    # -------------------------------------------------------------------------
    # XGBoost Quantile
    # -------------------------------------------------------------------------

    elif model_type == "xgb":

        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantile,
            **params,
        )

        model.fit(X_train, y_train)

        return model

    # -------------------------------------------------------------------------
    # GAM
    # -------------------------------------------------------------------------

    elif model_type == "gam":

        return


def predict_quantile(
    model,
    X: pd.DataFrame,
    quantile: float | None = None,
) -> np.ndarray:

    model_name = model.__class__.__name__

    if model_name == "RandomForestQuantileRegressor":

        if quantile is None:
            raise ValueError("quantile must be provided for RandomForestQuantileRegressor")

        return model.predict(X, quantiles=[quantile]).flatten() if not X.empty else []

    return model.predict(X) if not X.empty else []


def fit_q50_model(
    df: pd.DataFrame,
    features,
    issue_time,
    model_type,
    params=None
):
    """Dedicated workflow for q50 feature set."""

    QUANTILE = 0.5

    train_df, y_train_df, forecast_df = create_day_ahead_forecast_split(
        feature_df=df,
        features=features['q50'],
        issue_time=issue_time,
        target="Balancing_energy_total",
    )

    X_train = train_df
    y_train = y_train_df
    X_test = forecast_df[features['q50']]
    y_test = forecast_df["Balancing_energy_total"]

    if model_type == 'linear':
        X_train_clean, y_train_clean = remove_nan_from_training(X_train, y_train)
        model = fit_quantile_model(X_train_clean, y_train_clean, QUANTILE, model_type)
        prediction = predict_quantile(model, X=X_test)
        return model, prediction

    model = fit_quantile_model(X_train, y_train, QUANTILE, model_type, params)

    prediction = predict_quantile(model, X=X_test)

    df_pred = pd.DataFrame(index=y_test.index)
    df_pred['q50'] = prediction

    return model, df_pred


def fit_q95_model(
    df: pd.DataFrame,
    features,
    issue_time,
    model_type,
    params=None
):
    """Dedicated workflow for q50 feature set."""

    QUANTILE = 0.95

    train_df, y_train_df, forecast_df = create_day_ahead_forecast_split(
        feature_df=df,
        features=features['q95'],
        issue_time=issue_time,
        target="Balancing_energy_total",
    )

    X_train = train_df
    y_train = y_train_df
    X_test = forecast_df[features['q95']]
    y_test = forecast_df["Balancing_energy_total"]

    if model_type == 'linear':
        X_train_clean, y_train_clean = remove_nan_from_training(X_train, y_train)
        model = fit_quantile_model(X_train_clean, y_train_clean, QUANTILE, model_type)
        prediction = predict_quantile(model, X=X_test)
        return model, prediction

    model = fit_quantile_model(X_train, y_train, QUANTILE, model_type, params)

    prediction = predict_quantile(model, X=X_test)

    df_pred = pd.DataFrame(index=y_test.index)
    df_pred['q95'] = prediction

    return model, df_pred


def calculate_stress_metrics(
    q50: pd.DataFrame,
    q95: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate stress metrics from q50 and q95 forecasts.
    """

    stress_df = q50.merge(
        q95,
        left_index=True,
        right_index=True,
    )

    # Absolute tail spread
    stress_df["stress_range"] = stress_df["q95"] - stress_df["q50"]

    # Relative tail spread
    stress_df["stress_ratio"] = np.where(
        stress_df["q50"] != 0,
        stress_df["q95"] / stress_df["q50"],
        np.nan,
    )

    return stress_df


def remove_nan_from_training(X_train, y_train):

    # Count NaNs by column
    nan_counts = X_train.isna().sum().sort_values(ascending=False)
    nan_counts = nan_counts[nan_counts > 0]

    print("Columns containing NaNs:")
    print(nan_counts)

    print(f"\nTotal NaN values: {X_train.isna().sum().sum():,}")
    print(f"Rows containing at least one NaN: {X_train.isna().any(axis=1).sum():,}")
    print(f"Total rows: {len(X_train):,}")

    rows_with_nan = X_train.isna().any(axis=1)
    groups = rows_with_nan.ne(rows_with_nan.shift()).cumsum()
    nan_blocks = X_train.assign(has_nan=rows_with_nan, group=groups).query("has_nan").groupby("group").size()
    print("Print blocks:", nan_blocks)

    # Keep only rows with no NaNs in X
    mask = ~X_train.isna().any(axis=1)

    X_train_clean = X_train.loc[mask]
    y_train_clean = y_train.loc[mask]

    print(f"Original rows: {len(X_train):,}")
    print(f"Remaining rows: {len(X_train_clean):,}")
    print(f"Dropped rows: {len(X_train) - len(X_train_clean):,}")

    return X_train_clean, y_train_clean
