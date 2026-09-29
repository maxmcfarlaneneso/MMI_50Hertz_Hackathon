# =============================================================================
# WORKSTREAM 5: VALIDATION
# =============================================================================

import numpy as np
import pandas as pd

from sklearn.metrics import mean_pinball_loss


def _align_actuals_and_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> pd.DataFrame:
    """
    Align actual and predicted values and remove paired missing observations.
    """

    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)

    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(
            y_pred,
            index=y_true.index,
        )

    validation_data = pd.concat(
        [
            y_true.rename("y_true"),
            y_pred.rename("y_pred"),
        ],
        axis=1,
        join="inner",
    )

    if validation_data.empty:
        raise ValueError("No valid paired observations remain after alignment.")

    return validation_data


def pinball_loss(
    y_true: pd.Series,
    y_pred: pd.Series,
    quantile: float,
) -> float:
    """Quantile loss."""

    data = _align_actuals_and_predictions(
        y_true=y_true,
        y_pred=y_pred,
    )

    return float(
        mean_pinball_loss(
            y_true=data["y_true"],
            y_pred=data["y_pred"],
            alpha=quantile,
        )
    )


def reliability(
    mqr_reliability: pd.DataFrame,
    mqr_nominal: pd.Series,
    quantile_cols: list,
    quantiles: list,
):
    """
    Calculate the reliability diagram for a set of predictive quantiles.

    Parameters
    ----------
    mqr_reliability : pd.DataFrame
        DataFrame containing half-hourly predictive quantiles generated from
        an ensemble forecast. Each column represents a predictive quantile
        (e.g. ``"q5"``, ``"q10"``, ..., ``"q95"``), where the quantiles have been
        estimated from the distribution of 51 ensemble predictions for each
        half-hour settlement period. Values should be expressed on a 0–1 scale
        (e.g. an empirical exceedance probability or percentile rank).

    mqr_nominal : pd.Series
        Series containing the outturn (actual) demand values used as the
        reference distribution for evaluating reliability.

    quantile_cols : list[str]
        Names of the quantile columns in ``mqr_reliability``. These should be
        provided in string format matching the DataFrame columns, for example
        ``["q5", "q10", ..., "q95"]``.

    quantiles : list[float]
        Numerical quantile levels corresponding to ``quantile_cols``. These
        should be expressed between 0 and 1, for example
        ``[0.05, 0.10, ..., 0.95]``.

    Returns
    -------
    pd.Series
        Reliability values (in percent) indexed by quantile level
        (e.g. 5, 10, ..., 95). For a perfectly calibrated forecast, the
        reliability at the q-th quantile would be approximately q%.
    """

    nominal = pd.Series(
        {q_col: float(mqr_nominal.quantile(q)) for q, q_col in zip(quantiles, quantile_cols)}, name='nominal'
    ).loc[quantile_cols]

    mqr_reliability_unrav = mqr_reliability.values.ravel()

    reliability = pd.Series(
        [(1 - (sum(mqr_reliability_unrav > n) / len(mqr_reliability_unrav))) * 100 for n in nominal],
        index=map(lambda q: int(q[1:]), quantile_cols),
    )

    return reliability


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
