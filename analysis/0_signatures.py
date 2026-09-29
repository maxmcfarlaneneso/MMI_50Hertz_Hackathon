import pandas as pd
import numpy as np

# =============================================================================
# WORKSTREAM 1: DATA & FEATURE ENGINEERING
# =============================================================================

def read_data(
    data_dir: str = "./data",
) -> dict[str, pd.DataFrame]:
    """
    Read all source datasets.

    Returns
    -------
    dict[str, pd.DataFrame]

    Keys:
        balancing
        market
        load
        renewables_absolute
        renewables_metrics
        other_generation
        transmission
    """


# def assess_data_quality(
#     tables: dict[str, pd.DataFrame],
# ) -> pd.DataFrame:
#     """
#     Assess:

#     - coverage
#     - missingness
#     - duplicate timestamps
#     - gaps
#     - outliers

#     Returns summary table.
#     """


def add_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create common modelling features:

    - settlement_period
    - hour
    - day_of_week
    - month
    - is_weekend
    - day_type
    - tod_sin
    - tod_cos
    - dow_sin
    - dow_cos
    """


def add_candidate_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create engineered features including:

    Renewable composition
    ---------------------
    - wind_DA
    - renewables_DA
    - renewable_concentration_DA

    Ramp metrics
    ------------
    - abs_load_ramp_DA
    - abs_RES_ramp_DA
    - load_residual_ramp_DA
    - ramp_pressure_DA

    Availability metrics
    --------------------
    - UA_dispatchable
    - UA_total_to_load_DA
    - UA_dispatchable_to_load_DA
    - outage shares

    Cross-border metrics
    --------------------
    - abs_net_position_DA
    - net_position_to_load_DA
    - import_outage_pressure_DA

    Interaction terms
    -----------------
    - RES_share_x_UA_total
    - RES_share_x_import_dependency
    - residual_load_x_UA_total
    - etc.

    Ex-post diagnostics
    -------------------
    - load_forecast_error
    - won_forecast_error
    - wof_forecast_error
    - sol_forecast_error
    - renewables_forecast_error
    - residual_load_error
    - p_DA_to_imbalance_spread
    """


def prepare_candidate_feature_data(
    tables: dict[str, pd.DataFrame],
    target: str,
) -> tuple[
    pd.DataFrame,
    dict[str, list[str]],
]:
    """
    Create modelling dataframe and candidate
    feature groups.

    Returns
    -------
    featured_df
        Combined feature dataframe.

    feature_groups
        {
            "q50": [...],
            "q95": [...],
            "ex_post": [...],
        }
    """


def get_candidate_feature_sets(
    df: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Create candidate feature groups for:

    q50
        Expected balancing drivers.

    q95
        Extreme balancing drivers.

    ex_post
        Root-cause analysis using
        realised conditions and
        forecast errors.
    """


def reproduce_feature_set(
    version: int = 1,
) -> None:
    """
    Build and persist feature set.

    Saves:

        candidate_feature_data_v{version}.pickle
    """


def read_feature_set(
    version: int = 1,
) -> tuple[
    pd.DataFrame,
    dict[str, list[str]],
]:
    """
    Load previously generated feature set.

    Returns
    -------
    featured_df

    feature_groups
        {
            "q50": [...],
            "q95": [...],
            "ex_post": [...],
        }
    """

# =============================================================================
# WORKSTREAM 2: BENCHMARKS
# =============================================================================
#
# Forecasting convention
# ----------------------
#
# Issue time:
#     18:00 on D-1
#
# Forecast horizon:
#     All settlement periods on D
#
# Information available:
#     Only observations strictly before
#     issue_time.
#
# Example
# -------
#
# issue_time = 2026-01-01 18:00
#
# forecast horizon =
#
#     2026-01-02 SP01
#     2026-01-02 SP02
#     ...
#     2026-01-02 SP96
#
# Training information:
#
#     df.index < issue_time
#
# Output:
#
#     pd.Series indexed by all timestamps
#     in D (the next day)
#


def benchmark_zero(
    train_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target: str,
) -> pd.Series:
    """
    Predict zero for every settlement period
    on D.

    Parameters
    ----------
    train_df
        Historical observations available
        before issue time.

    forecast_df
        Settlement periods to forecast
        (all SPs on D).

    target
        Target variable.

    Returns
    -------
    pd.Series

        index = forecast_df.index

        values = 0
    """


def benchmark_previous_day(
    train_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target: str,
    lag_days: int = 1,
) -> pd.Series:
    """
    Forecast each SP on D using the same
    settlement period from D-1.

    Example

        Forecast:
            D SP37

        Uses:
            D-1 SP37

    Returns
    -------
    pd.Series

        index = forecast_df.index
    """


def benchmark_average_by_settlement_period(
    train_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target: str,
    sp_col: str = "settlement_period",
) -> pd.Series:
    """
    Historical climatology by settlement
    period.

    Example

        Forecast:
            D SP37

        Uses:

            mean(
                target
                for all historic SP37
                observations in train_df
            )

    Returns
    -------
    pd.Series

        index = forecast_df.index
    """


def benchmark_average_by_hour_daytype(
    train_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target: str,
    hour_col: str = "hour",
    daytype_col: str = "day_type",
) -> pd.Series:
    """
    Historical climatology by hour and
    day type.

    Example

        Forecast:
            Weekend 16:30

        Uses:

            mean(
                target
                where:
                    hour == 16
                    day_type == weekend
                in train_df
            )

    Returns
    -------
    pd.Series

        index = forecast_df.index
    """


def benchmark_average_by_sp_daytype(
    train_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target: str,
    sp_col: str = "settlement_period",
    daytype_col: str = "day_type",
) -> pd.Series:
    """
    Historical climatology by settlement
    period and day type.

    Example

        Forecast:
            Weekend SP37

        Uses:

            mean(
                target
                where:
                    settlement_period == 37
                    day_type == weekend
                in train_df
            )

    Returns
    -------
    pd.Series

        index = forecast_df.index
    """


def create_forecast_split(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into:

        train_df
            observations strictly before
            issue_time

        forecast_df
            all settlement periods on D
            (the day following issue_time)

    Example

        issue_time:
            2026-01-01 18:00

        train_df:
            df[df.index < issue_time]

        forecast_df:
            all rows for 2026-01-02
    """


def run_benchmark_suite(
    train_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """
    Generate benchmark forecasts for the
    full next day.

    Returns
    -------
    pd.DataFrame

        index:
            all settlement periods on D

        columns:

            zero
            previous_day
            climatology_sp
            climatology_hour_daytype
            climatology_sp_daytype
    """
# =============================================================================
# WORKSTREAM 3: QUANTILE MODELLING
# =============================================================================

def fit_quantile_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantile: float,
    model_type: str,
):
    """
    model_type:
    - xgb
    - qrf
    - linear
    - gam
    """


def predict_quantile(
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    """Generate quantile predictions."""


def fit_q50_model(
    df: pd.DataFrame,
):
    """Dedicated workflow for q50 feature set."""


def fit_q95_model(
    df: pd.DataFrame,
):
    """Dedicated workflow for q95 feature set."""


def calculate_stress_metrics(
    q50: pd.Series,
    q95: pd.Series,
) -> pd.DataFrame:
    """
    q95_minus_q50
    q95_div_q50
    """
    

# =============================================================================
# WORKSTREAM 4: UNCERTAINTY ASSESSMENT
# =============================================================================

def bootstrap_quantile_model(
    X: pd.DataFrame,
    y: pd.Series,
    quantile: float,
    model_type: str,
    n_boot: int = 100,
):
    """Fit bootstrap ensemble."""


def estimate_bootstrap_uncertainty(
    bootstrap_predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Mean
    Std deviation
    Confidence intervals
    """


def compare_learning_methods(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    quantile: float,
) -> pd.DataFrame:
    """
    Compare:
    - XGBoost Quantile
    - Quantile Random Forest
    - Linear Quantile
    - GAM
    """


def classify_uncertainty_source(
    q_spread: pd.Series,
    model_spread: pd.Series,
    bootstrap_spread: pd.Series,
) -> pd.Series:
    """
    Aleatoric vs epistemic assessment.
    """
    

# =============================================================================
# WORKSTREAM 5: VALIDATION
# =============================================================================

def pinball_loss(
    y_true: pd.Series,
    y_pred: pd.Series,
    quantile: float,
) -> float:
    """Quantile loss."""


def calculate_quantile_coverage(
    y_true: pd.Series,
    q_pred: pd.Series,
) -> float:
    """
    Fraction of observations below forecast quantile.
    """


def calibration_table(
    y_true: pd.Series,
    quantile_predictions: dict,
) -> pd.DataFrame:
    """
    Coverage statistics for multiple quantiles.
    """


def reliability_curve(
    y_true: pd.Series,
    quantile_predictions: dict,
) -> pd.DataFrame:
    """Data for reliability/calibration plots."""


def evaluate_probabilistic_model(
    y_true: pd.Series,
    quantile_predictions: pd.DataFrame,
) -> dict:
    """
    Pinball loss
    Coverage
    Calibration
    """
    

# =============================================================================
# WORKSTREAM 6: EXPLAINABILITY
# =============================================================================

def calculate_shap_values(
    model,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """SHAP attribution values."""


def rank_feature_importance(
    shap_values: pd.DataFrame,
) -> pd.DataFrame:
    """Global importance ranking."""


def compare_quantile_drivers(
    shap_q50: pd.DataFrame,
    shap_q95: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify features with
    disproportionate impact on q95.
    """


def identify_tail_risk_drivers(
    shap_q50: pd.DataFrame,
    shap_q95: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank variables by:

        importance(q95)
        -
        importance(q50)

    """
    

# =============================================================================
# FINAL PIPELINE
# =============================================================================

def run_hackathon_pipeline(
    raw_data: pd.DataFrame,
) -> dict:
    """
    1. Feature engineering
    2. Benchmarks
    3. Quantile models
    4. Validation
    5. Uncertainty assessment
    6. Explainability

    Returns all artefacts required
    for the final presentation.
    """