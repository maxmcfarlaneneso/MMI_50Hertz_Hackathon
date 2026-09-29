# GENERATE DETERMANISTIC MODELS TO VALIDATE USE OF PROBABILISTIC FRAMEWORK FOR FORECASTING
#GET DATA

import pandas as pd
import pyarrow as pa
import numpy as np

#Model Functions 


#Forecast split
def create_forecast_split(
    df,
    issue_time,
    target):

    df.index = pd.to_datetime(
        df.index)

    if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("feature_df.index must be a DatetimeIndex.")
    
    issue_time = pd.Timestamp(issue_time)
    
    if issue_time.hour != 18 or issue_time.minute != 0:
        raise ValueError("issue_time must be 18:00 on D-1.")
    
    forecast_date = issue_time.normalize() + pd.Timedelta(days=1)
    forecast_end = forecast_date + pd.Timedelta(days=1)
    
    train_df = df.loc[df.index < issue_time].copy()
    
    forecast_df = df.loc[
            (df.index >= forecast_date)
            & (df.index < forecast_end)
        ].copy()
    
    if forecast_df.empty:
        raise ValueError(
                f"No delivery rows found for forecast date "
                f"{forecast_date.date()}."
            )
    
    if target in forecast_df.columns:
            # Hide the realised target from the forecasting interface.
            forecast_df[target] = np.nan
    
    return train_df, forecast_df



#zero benchmark
def benchmark_zero(
    train_df,
    forecast_df,
    target):

    return pd.Series(
        0.0,
        index=forecast_df.index,
        name=target)


#Previous Day 
def benchmark_prev(
    train_df,
    forecast_df,
    target):

    n_periods = len(forecast_df)

    if len(train_df) < n_periods:
        raise ValueError(
            f"Training data has {len(train_df)} rows, "
            f"but {n_periods} previous observations are required."
        )

    return pd.Series(
        data=train_df[target].iloc[-n_periods:].to_numpy(),
        index=forecast_df.index,
        name=target,
        dtype=float)



#SP Avg
def benchmark_avg_sp(
    train_df,
    forecast_df,
    target):
    """Forecast using the historical mean for each settlement period."""

    settlement_period_mean = (
        train_df
        .groupby("settlement_period")[target]
        .mean()
    )

    forecast_values = (
        forecast_df["settlement_period"]
        .map(settlement_period_mean)
        .to_numpy()
    )

    return pd.Series(
        data=forecast_values,
        index=forecast_df.index,
        name=target,
        dtype=float
    )



#SP & Weekday AVG
def benchmark_avg_sp_weekday(
    train_df,
    forecast_df,
    target,
):

    profile = (
        train_df
        .groupby(
            ["settlement_period", "dow"]
        )[target]
        .mean()
    )

    forecast_values = (
        forecast_df.set_index(
            ["settlement_period", "dow"]
        ).index.map(profile)
    )

    return pd.Series(
        data=forecast_values,
        index=forecast_df.index,
        name=target,
        dtype=float
    )


#Run suite
def run_benchmark_suite(
    train_df,
    forecast_df,
    target
):

    return {
        "b_zero": benchmark_zero(
            train_df,
            forecast_df,
            target
        ),
        "b_prevday": benchmark_prev(
            train_df,
            forecast_df,
            target
        ),
        "b_avg_sp": benchmark_avg_sp(
            train_df,
            forecast_df,
            target
        ),
        "b_avg_sp_weekday": benchmark_avg_sp_weekday(
            train_df,
            forecast_df,
            target
        )
    }


#Benchmarking errors

def benchmarking_errors(
    raw,
    issue_times,
    target
):

    errors_dict = {}

    for issue_time in issue_times:

        train, forecast = create_forecast_split(
            raw,
            issue_time,
            target
        )

        benchmark_dict = run_benchmark_suite(
            train,
            forecast,
            target
        )

        benchmark_errors = {}

        for key, value in benchmark_dict.items():

            # Each benchmark returns a Series, so convert it
            # into a one-column DataFrame.
            df_to_merge = value.to_frame()

            df_error = pd.merge(
                df_to_merge,
                raw[[target]],
                left_index=True,
                right_index=True,
                how="left",
                suffixes=[
                    " Forecast",
                    " Actual"
                ]
            )

            df_error["Benchmark Error"] = (
                df_error[f"{target} Actual"]
                - df_error[f"{target} Forecast"]
            )

            df_error["Benchmark"] = key

            benchmark_errors[key] = df_error

        errors_dict[issue_time] = pd.concat(
            benchmark_errors.values()
        )

    errors = pd.concat(
        [
            df.assign(IssueTime=issue_time)
            for issue_time, df in errors_dict.items()
        ]
    )

    return errors

    

              

            






#Import
dep = pd.read_feather(
    r".\data\1_Target variable_balancing.feather"
)

features = pd.read_parquet(
    r".\data\datetime_features_15min.parquet"
)


#Config 
bz = "FR"
issue_time = pd.to_datetime("2026-01-01 18:00:00")

features.index = pd.to_datetime(
    features.index).strftime(
        "%Y-%m-%d %H:%M:%S")

dep = dep.set_index("Datetime")

dep = dep.merge(features[["settlement_period","dow"]],
right_index=True,
left_index=True,
how="left")

dep = dep[["settlement_period",
           "dow",
           "Balancing_energy_total",
           "BZ"]]

dep = dep[dep["BZ"]=="FR"]

dep.index = pd.to_datetime(
    dep.index).strftime(
        "%Y-%m-%d %H:%M:%S")


#Run Functions with test data

train_df, forecast_df = create_forecast_split(
     dep,
     issue_time,
     "balancing_energy_total"
)


test_zero = benchmark_zero(
    train_df,
    forecast_df,
    "Balancing_energy_total"
)



test_prev = benchmark_prev(
    train_df,
    forecast_df,
    "Balancing_energy_total"
)



test_sp = benchmark_avg_sp(
    train_df,
    forecast_df,
    "Balancing_energy_total"
)


test_sp_weekday = benchmark_avg_sp_weekday(
    train_df,
    forecast_df,
    "Balancing_energy_total"
)


all_benchmarks = run_benchmark_suite(
    train_df,
    forecast_df,
    "Balancing_energy_total"
)

issue_times = pd.date_range("2024-01-01 18:00",
                            "2025-01-01 18:00",
                            freq="1D")

errors_df = benchmarking_errors(
     dep,
     issue_times,
     "Balancing_energy_total") 

print(errors_df.head())