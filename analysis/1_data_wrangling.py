from __future__ import annotations

from typing import Mapping
import os

from pathlib import Path
import numpy as np
import pandas as pd
import pickle 

from pathlib import Path
import pickle
import re


# =============================================================================
# CONFIGURATION
# =============================================================================

KEY_COLS = ["Datetime", "BZ"]
TARGET = "Balancing_energy_total"

FR_NEIGHBOURS = [
    "BE",      # Belgium
    "DE_LU",   # Germany-Luxembourg
    "CH",      # Switzerland
    "IT-NORTH",# Italy North
    "ES",      # Spain
]

# Variables that are potentially available when the forecast is issued
# at 18:00 on D-1.
DAY_AHEAD_RAW_FEATURES = [
    # Calendar variables
    "Season",
    "Weekday",
    "Hour_of_day",

    # Market
    "p_SDAC",

    # Demand
    "load_DA",
    "load_ramp_DA",

    # Renewables
    "won_DA",
    "wof_DA",
    "sol_DA",
    "RES_share_DA",
    "load_residual_DA",
    "RES_ramp_DA",

    # Generation availability
    "UA_total",
    "UA_thermal",
    "UA_nuclear",
    "UA_hydro",

    # Transmission
    "Net_pos_DA",
    "Net_exp_ramps_DA",
    "import_depend_DA",
]

# Explicitly excluded from a day-ahead forecast issued at 18:00 on D-1.
# These variables are useful for ex-post analysis or intraday models only.
EX_POST_OR_INTRADAY_FEATURES = [
    "Balancing_energy_up",
    "Balancing_energy_down",
    "Imbalance",
    "p_IDA1",
    "p_IDA2",
    "p_IDA3",
    "p_imbalance",
    "load_act",
    "load_ramp_act",
    "won_ID",
    "won_act",
    "wof_ID",
    "wof_act",
    "sol_ID",
    "sol_act",
    "RES_share_act",
    "load_residual_act",
    "RES_ramp_act",
    "Net_pos_ID",
    "Net_exp_ramps_ID",
    "import_depend_ID",
]


# =============================================================================
# VALIDATION AND MERGING
# =============================================================================

def validate_input_table(
    df: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:
    """
    Validate and standardise one source table.

    Required columns
    ----------------
    Datetime
        Delivery timestamp.
    BZ
        Bidding-zone identifier.

    Returns
    -------
    pd.DataFrame
        Copy with a parsed Datetime column and duplicate keys removed.
    """
    missing = set(KEY_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{table_name!r} is missing required columns: {sorted(missing)}"
        )

    out = df.copy()
    out["Datetime"] = pd.to_datetime(out["Datetime"], errors="raise")

    duplicate_mask = out.duplicated(KEY_COLS, keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            f"{table_name!r} contains {duplicate_count} rows with duplicate "
            f"(Datetime, BZ) keys."
        )

    return out.sort_values(KEY_COLS)


def merge_source_tables(
    tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge the seven source tables on Datetime and BZ.

    Duplicate calendar columns such as Season, Weekday, and Hour_of_day are
    retained from the first table and omitted from subsequent tables.

    Parameters
    ----------
    tables
        Mapping of table name to DataFrame.

    Returns
    -------
    pd.DataFrame
        Datetime-indexed panel with one row per timestamp and bidding zone.
    """
    if not tables:
        raise ValueError("No source tables were supplied.")

    merged = None

    for table_name, raw_df in tables.items():
        current = validate_input_table(raw_df, table_name)

        if merged is None:
            merged = current
            continue

        repeated_context_cols = [
            col
            for col in ["Season", "Weekday", "Hour_of_day"]
            if col in current.columns and col in merged.columns
        ]

        current = current.drop(columns=repeated_context_cols)

        merged = merged.merge(
            current,
            on=KEY_COLS,
            how="outer",
            validate="one_to_one",
        )

    if TARGET not in merged.columns:
        raise ValueError(f"Target column {TARGET!r} was not found.")

    return (
        merged
        .sort_values(KEY_COLS)
        .set_index("Datetime")
    )



# =============================================================================
# DERIVED DAY-AHEAD FEATURES
# =============================================================================

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide while replacing zero denominators and infinities with NaN."""
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def add_candidate_features(
    df: pd.DataFrame,
    target_bz
) -> pd.DataFrame:

    price_differentials = df.reset_index().pivot(index='Datetime', columns='BZ', values='p_SDAC').add_suffix('_p_SDAC')

    df = df.query(f'BZ=="{target_bz}"')
    df = df.merge(
        price_differentials,
        on="Datetime",
        how="left",
    )
    renewables_DA = (
        df[["won_DA", "wof_DA", "sol_DA"]]
        .sum(axis=1, min_count=1)
    )

    renewables_act = (
        df[["won_act", "wof_act", "sol_act"]]
        .sum(axis=1, min_count=1)
    )

    load_residual_ramp_DA = (
        df.groupby("BZ", sort=False)["load_residual_DA"]
        .diff()
    )
    # target_prices = (
    #     df.loc[df["BZ"] == target_bz, ["Datetime", "p_SDAC"]]
    #     .rename(columns={"p_SDAC": "p_SDAC_target"})
    # )

    # df = df.merge(
    #     target_prices,
    #     on="Datetime",
    #     how="left",
    # )
    FR_NEIGHBOURS = [
        "BE",
        "DE_LU",
        "CH",
        "IT-NORTH",
        "ES",
    ]

    p_SDAC_diff_cols = []

    for bz in FR_NEIGHBOURS:
        col = f"FR_{bz}_price_spread"
        p_SDAC_diff_cols.append(col)
        df[col] = (
            df["p_SDAC"]
            - df[f"{bz}_p_SDAC"]
        )

    return (
        df
        .assign(
            # -----------------------------------------------------------------
            # Renewable composition
            # -----------------------------------------------------------------

            wind_DA=lambda x: (
                x["won_DA"] + x["wof_DA"]
            ),

            renewables_DA=renewables_DA,

            renewables_act=renewables_act,

            onshore_wind_fraction_DA=lambda x: safe_divide(
                x["won_DA"],
                renewables_DA,
            ),

            offshore_wind_fraction_DA=lambda x: safe_divide(
                x["wof_DA"],
                renewables_DA,
            ),

            solar_fraction_DA=lambda x: safe_divide(
                x["sol_DA"],
                renewables_DA,
            ),

            # -----------------------------------------------------------------
            # Renewable concentration
            # -----------------------------------------------------------------

            renewable_concentration_DA=lambda x: (
                x["onshore_wind_fraction_DA"].pow(2)
                + x["offshore_wind_fraction_DA"].pow(2)
                + x["solar_fraction_DA"].pow(2)
            ),

            # -----------------------------------------------------------------
            # Ramp metrics
            # -----------------------------------------------------------------

            abs_load_ramp_DA=lambda x: x["load_ramp_DA"].abs(),

            abs_RES_ramp_DA=lambda x: x["RES_ramp_DA"].abs(),

            abs_net_export_ramp_DA=lambda x: (
                x["Net_exp_ramps_DA"].abs()
            ),

            load_residual_ramp_DA=load_residual_ramp_DA,

            abs_load_residual_ramp_DA=lambda x: (
                x["load_residual_ramp_DA"].abs()
            ),

            ramp_pressure_DA=lambda x: (
                x["abs_load_ramp_DA"].fillna(0)
                + x["abs_RES_ramp_DA"].fillna(0)
                + x["abs_net_export_ramp_DA"].fillna(0)
            ),

            # -----------------------------------------------------------------
            # Availability / outages
            # -----------------------------------------------------------------

            UA_dispatchable=lambda x: (
                x["UA_thermal"]
                + x["UA_nuclear"]
                + x["UA_hydro"]
            ),

            UA_total_to_load_DA=lambda x: safe_divide(
                x["UA_total"],
                x["load_DA"].abs(),
            ),

            UA_thermal_to_load_DA=lambda x: safe_divide(
                x["UA_thermal"],
                x["load_DA"].abs(),
            ),

            UA_nuclear_to_load_DA=lambda x: safe_divide(
                x["UA_nuclear"],
                x["load_DA"].abs(),
            ),

            UA_dispatchable_to_load_DA=lambda x: safe_divide(
                x["UA_dispatchable"],
                x["load_DA"].abs(),
            ),

            UA_thermal_share=lambda x: safe_divide(
                x["UA_thermal"],
                x["UA_total"],
            ),

            UA_nuclear_share=lambda x: safe_divide(
                x["UA_nuclear"],
                x["UA_total"],
            ),

            UA_hydro_share=lambda x: safe_divide(
                x["UA_hydro"],
                x["UA_total"],
            ),

            # -----------------------------------------------------------------
            # Transmission / cross-border
            # -----------------------------------------------------------------

            abs_net_position_DA=lambda x: (
                x["Net_pos_DA"].abs()
            ),

            net_position_to_load_DA=lambda x: safe_divide(
                x["Net_pos_DA"],
                x["load_DA"].abs(),
            ),

            abs_net_position_to_load_DA=lambda x: (
                x["net_position_to_load_DA"].abs()
            ),

            import_outage_pressure_DA=lambda x: (
                x["import_depend_DA"]
                * x["UA_dispatchable_to_load_DA"]
            ),

            # -----------------------------------------------------------------
            # Interaction terms
            # -----------------------------------------------------------------

            RES_share_x_UA_total=lambda x: (
                x["RES_share_DA"] * x["UA_total"]
            ),

            RES_share_x_UA_thermal=lambda x: (
                x["RES_share_DA"] * x["UA_thermal"]
            ),

            RES_share_x_UA_nuclear=lambda x: (
                x["RES_share_DA"] * x["UA_nuclear"]
            ),

            residual_load_x_UA_total=lambda x: (
                x["load_residual_DA"] * x["UA_total"]
            ),

            residual_load_x_UA_thermal=lambda x: (
                x["load_residual_DA"] * x["UA_thermal"]
            ),

            RES_ramp_x_UA_total=lambda x: (
                x["abs_RES_ramp_DA"] * x["UA_total"]
            ),

            load_ramp_x_UA_total=lambda x: (
                x["abs_load_ramp_DA"] * x["UA_total"]
            ),

            residual_ramp_x_UA_total=lambda x: (
                x["abs_load_residual_ramp_DA"]
                * x["UA_total"]
            ),

            RES_share_x_import_dependency=lambda x: (
                x["RES_share_DA"]
                * x["import_depend_DA"]
            ),

            residual_load_x_import_dependency=lambda x: (
                x["load_residual_DA"]
                * x["import_depend_DA"]
            ),

            price_x_residual_load=lambda x: (
                x["p_SDAC"]
                * x["load_residual_DA"]
            ),

            price_x_UA_total=lambda x: (
                x["p_SDAC"]
                * x["UA_total"]
            ),

            # -----------------------------------------------------------------
            # Ex-post forecast errors
            # -----------------------------------------------------------------

            load_forecast_error=lambda x: (
                x["load_act"] - x["load_DA"]
            ),

            won_forecast_error=lambda x: (
                x["won_act"] - x["won_DA"]
            ),

            wof_forecast_error=lambda x: (
                x["wof_act"] - x["wof_DA"]
            ),

            sol_forecast_error=lambda x: (
                x["sol_act"] - x["sol_DA"]
            ),

            renewables_forecast_error=lambda x: (
                renewables_act - renewables_DA
            ),

            residual_load_error=lambda x: (
                x["load_residual_act"]
                - x["load_residual_DA"]
            ),

            # -----------------------------------------------------------------
            # Ex-post price metrics
            # -----------------------------------------------------------------

            p_DA_to_imbalance_spread=lambda x: (
                x["p_imbalance"] - x["p_SDAC"]
            ),

            p_IDA1_to_DA_spread=lambda x: (
                x["p_IDA1"] - x["p_SDAC"]
            ),

            p_IDA3_to_DA_spread=lambda x: (
                x["p_IDA3"] - x["p_SDAC"]
            ),
        )
    )

# =============================================================================
# CANDIDATE FEATURE SETS
# =============================================================================

def get_candidate_feature_sets(
    df: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Return available candidate predictors for q50 and q95 models.

    The feature sets deliberately overlap. The question is not whether a
    driver affects only one quantile, but whether the magnitude and form of
    its effect differs between expected and upper-tail balancing volumes.
    """

        
    calendar_features = [
        "tod_sin",
        "tod_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "month",
        "settlement_period",
    ]

    q50_features = [
        # Core demand and system state
        "abs_load_residual_ramp_DA",
        "wof_DA",
        "won_DA",
        "p_SDAC",
        "UA_thermal",
        "UA_nuclear",
        "renewable_concentration_DA",


        "load_DA",
        "load_ramp_DA",
        "load_residual_DA",
        "RES_share_DA",

        # Availability
        # "UA_total",
        "UA_hydro",
        "UA_dispatchable_to_load_DA",

        # Market and cross-border position
        "Net_pos_DA",
        "import_depend_DA",

        'FR_BE_price_spread', 'FR_DE_LU_price_spread', 'FR_CH_price_spread', 'FR_IT-NORTH_price_spread', 'FR_ES_price_spread',

        # Calendar
        *calendar_features,
    ]

    q95_features = [
        # Retain the core q50 system drivers
        "load_DA",
        "load_residual_DA",
        "RES_share_DA",
        "p_SDAC",

        # Renewable composition and variability
        "won_DA",
        "wof_DA",
        "sol_DA",
        "renewables_DA",
        "renewable_concentration_DA",
        "abs_load_ramp_DA",
        "abs_RES_ramp_DA",
        "abs_load_residual_ramp_DA",
        "ramp_pressure_DA",

        # Outage and availability pressure
        "UA_total",
        "UA_thermal",
        "UA_nuclear",
        "UA_hydro",
        "UA_dispatchable",
        "UA_total_to_load_DA",
        "UA_thermal_to_load_DA",
        "UA_nuclear_to_load_DA",
        "UA_dispatchable_to_load_DA",

        # Cross-border dependence
        "Net_pos_DA",
        "Net_exp_ramps_DA",
        "import_depend_DA",
        "abs_net_position_to_load_DA",
        "import_outage_pressure_DA",

        'FR_BE_price_spread', 'FR_DE_LU_price_spread', 'FR_CH_price_spread', 'FR_IT-NORTH_price_spread', 'FR_ES_price_spread',
        # # Compound-risk interactions
        # "RES_share_x_UA_total",
        # "RES_share_x_UA_thermal",
        # "RES_share_x_UA_nuclear",
        # "residual_load_x_UA_total",
        # "residual_load_x_UA_thermal",
        # "RES_ramp_x_UA_total",
        # "load_ramp_x_UA_total",
        # "residual_ramp_x_UA_total",
        # "RES_share_x_import_dependency",
        # "residual_load_x_import_dependency",
        # "price_x_residual_load",
        # "price_x_UA_total",

        # Calendar
        *calendar_features,
    ]
    q50_features = ['abs_load_residual_ramp_DA', 'FR_CH_price_spread', 'wof_DA', 'settlement_period', 'UA_nuclear', 'p_SDAC', 'UA_hydro', 'load_ramp_DA', 'RES_share_DA', 'UA_thermal']
    q95_features = ['FR_CH_price_spread', 'abs_load_residual_ramp_DA', 'UA_thermal', 'UA_hydro', 'p_IDA3', 'month', 'load_residual_DA', 'load_act', 'RES_share_DA', 'FR_IT-NORTH_price_spread']
    ex_post_features = [
        # ---------------------------------------------------------------------
        # Forecast errors
        # ---------------------------------------------------------------------

        "load_DA",
        "load_act",
        "load_forecast_error",

        "won_DA",
        "won_act",
        "won_forecast_error",

        "wof_DA",
        "wof_act",
        "wof_forecast_error",

        "sol_DA",
        "sol_act",
        "sol_forecast_error",

        "renewables_DA",
        "renewables_act",
        "renewables_forecast_error",

        # ---------------------------------------------------------------------
        # Actual system conditions
        # ---------------------------------------------------------------------

        "load_residual_DA",
        "load_residual_act",
        "residual_load_error",

        "RES_share_DA",
        "RES_share_act",

        "load_ramp_DA",
        "load_ramp_act",

        "RES_ramp_DA",
        "RES_ramp_act",

        # ---------------------------------------------------------------------
        # Market outcomes
        # ---------------------------------------------------------------------

        "p_SDAC",
        "p_IDA1",
        "p_IDA2",
        "p_IDA3",
        "p_imbalance",

        "p_DA_to_imbalance_spread",

        # ---------------------------------------------------------------------
        # Availability
        # ---------------------------------------------------------------------

        "UA_total",
        "UA_thermal",
        "UA_nuclear",
        "UA_hydro",

        # ---------------------------------------------------------------------
        # Cross-border conditions
        # ---------------------------------------------------------------------

        "Net_pos_DA",
        "Net_pos_ID",

        "import_depend_DA",
        "import_depend_ID",

        "Net_exp_ramps_DA",
        "Net_exp_ramps_ID",

        # ---------------------------------------------------------------------
        # Actual imbalance metrics
        # ---------------------------------------------------------------------

        "Imbalance",

        # ---------------------------------------------------------------------
        # Calendar
        # ---------------------------------------------------------------------

        *calendar_features,
    ]
    # Return only columns that were successfully constructed.
    return {
        "q50": [col for col in q50_features if col in df.columns],
        "q95": [col for col in q95_features if col in df.columns],
        "ex_post": [col for col in ex_post_features if col in df.columns],
    }

# =============================================================================
# D-1 18:00 FORECAST SPLIT
# =============================================================================

def create_day_ahead_forecast_split(
    feature_df: pd.DataFrame,
    features: list,
    issue_time: pd.Timestamp,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the history and full-day forecast frames for one issue time.

    Forecasting convention
    ----------------------
    Issue time
        18:00 on D-1.

    Training history
        Rows with delivery timestamps strictly before issue_time.

    Forecast horizon
        Every available settlement period on D, where D is the calendar day
        immediately following the issue date.

    Important
    ---------
    This function enforces the delivery-time cut-off for observed targets.
    It cannot verify when each source value was first published. Therefore,
    DA and outage fields must represent the vintage genuinely available by
    18:00 on D-1, rather than revised or ex-post values.
    """
    if not isinstance(feature_df.index, pd.DatetimeIndex):
        raise TypeError("feature_df.index must be a DatetimeIndex.")

    issue_time = pd.Timestamp(issue_time)

    if issue_time.hour != 18 or issue_time.minute != 0:
        raise ValueError("issue_time must be 18:00 on D-1.")

    forecast_date = issue_time.normalize() + pd.Timedelta(days=1)
    forecast_end = forecast_date + pd.Timedelta(days=1)

    train_df = feature_df.loc[feature_df.index < issue_time, features].copy()
    y_train_df = feature_df.loc[feature_df.index < issue_time, target].copy()

    forecast_df = feature_df.loc[
        (feature_df.index >= forecast_date)
        & (feature_df.index < forecast_end)
    ].copy()

    if forecast_df.empty:
        raise ValueError(
            f"No delivery rows found for forecast date "
            f"{forecast_date.date()}."
        )

    if target in forecast_df.columns:
        # Hide the realised target from the forecasting interface.
        forecast_df[target] = np.nan

    return train_df, y_train_df, forecast_df


def add_calendar_features(data_in):
    datetimes_ = pd.read_parquet(r'data\raw\datetime_features_15min.parquet')
    data_out = data_in.merge(
        datetimes_, left_index=True, right_index=True
    )
    return data_out


# =============================================================================
# END-TO-END PREPARATION
# =============================================================================

def prepare_candidate_feature_data(
    tables: Mapping[str, pd.DataFrame],
    target: str = TARGET,
    BZ='FR'
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, list[str]],
]:
    """
    Merge source data, engineer features, and create a D-1 18:00 split.

    Returns
    -------
    train_df
        Historical rows strictly before 18:00 on D-1.

    forecast_df
        All available settlement periods on D, with the target hidden.

    candidate_features
        Dictionary containing separate q50 and q95 candidate feature lists.
    """
    merged = merge_source_tables(tables)
    featured = add_calendar_features(merged)
    featured = add_candidate_features(featured, target_bz=BZ)

    candidate_features = get_candidate_feature_sets(featured)

    candidate_features['response'] = [target]

    featured_filtered = featured.loc[
        :, list(set(sum(list(candidate_features.values()), [])))
    ]
    return featured_filtered, candidate_features


def read_data(BZ='FR') -> dict[str, pd.DataFrame]:
    data_ = {}

    for f in Path("./data/raw").glob("*.feather"):
        df = pd.read_feather(f)

        # if BZ is not None:
        #     df = df.query(f'BZ=="{BZ}"')

        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)

        data_[os.path.splitext(f.name)[0]] = df

    return data_


def reproduce_feature_set(version = 6, BZ='FR'):
    data_ = read_data(BZ=BZ)

    featured_filtered, candidate_features = (
        prepare_candidate_feature_data(
            tables=data_,
            target="Balancing_energy_total",
            BZ=BZ
        )
    )

    featured_filtered = featured_filtered.assign(BZ=BZ)    
    featured_filtered = featured_filtered.assign(version=version)    

    output_path = f'./data/processed/candidate_feature_data_v{version}.pickle'

    with open(output_path, 'wb+') as f: 
        pickle.dump((featured_filtered, candidate_features), f)


def read_feature_set(
    version: int | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Load a previously generated candidate feature set.

    Parameters
    ----------
    version
        Feature set version number.

        If None, the latest available version
        is loaded automatically.

    Returns
    -------
    featured_df
        Feature dataframe containing engineered
        features.

    candidate_features
        Dictionary containing:

            {
                "q50": [...],
                "q95": [...],
                "ex_post": [...],
            }
    """

    data_dir = Path("./data/processed")

    if version is None:

        input_path = max(data_dir.glob("candidate_feature_data_v*.pickle"))

    else:

        input_path = (
            data_dir /
            f"candidate_feature_data_v{version}.pickle"
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Feature set not found: {input_path}"
            )

    with open(input_path, "rb") as f:
        featured_df, candidate_features = pickle.load(f)

    return featured_df, candidate_features

if __name__ == "__main__":

    reproduce_feature_set(BZ='FR')

    featured_filtered, candidate_features = (
        read_feature_set()
    )
    print