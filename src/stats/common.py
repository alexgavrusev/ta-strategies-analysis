import numpy as np
import pandas as pd
import scipy.stats as stats

# region Sharpe ratio


def estimated_sharpe_ratio(
    returns: pd.Series,
) -> np.float64:
    """
    Calculate the estimated Sharpe ratio, assuming 0 as
    the risk-free rate

    .. math::
        \\widehat{\\text{SR}} =
            \\frac{\\hat{\\mu}}{\\hat{\\sigma}}

    Parameters
    ----------
    returns : pd.Series
        A series of returns of the strategy, in the
        original sampling frequency

    Returns
    -------
    np.float64
    The estimated Sharpe ratio,
    calculated as the mean divided by the standard deviation
    """
    mean = returns.mean()
    std = returns.std(ddof=1)

    # all-zeroes series
    if mean == 0 and std == 0:
        return np.float64(0)

    sr = mean / std

    return sr


def annualized_estimated_sharpe_ratio(returns: pd.Series, periods: int) -> np.float64:
    """
    Calculate the annualized version of the
    estimated Sharpe ratio,
    assuming 0 as the risk-free rate.

    The Sharpe ratio will be adjusted for autocorrelation,
    following the method outlined in
    "The Statistics of Sharpe Ratios" (Lo, 2002)

    Parameters
    ----------
    returns : pd.Series
        A series of returns of the strategy,
        in the original sampling frequency
    periods : int
        Frequency of the returns,
        e.g. 52 when weekly returns are provided

    Returns
    -------
    np.float64
    The annualized estimated Sharpe ratio
    """
    sharpe_ratio = estimated_sharpe_ratio(returns)

    # might be an all-zeroes returns series,
    # which would produce NaN autocorr
    if sharpe_ratio == 0:
        return np.float64(0)

    annual_multiplier = np.sqrt(periods)

    rho = returns.autocorr(lag=1)

    # part of scale factor in equation 22
    autocorr_multiplier = (
        1 + (2 * rho / (1 - rho)) * (1 - ((1 - rho**periods) / (periods * (1 - rho))))
    ) ** (-0.5)

    return annual_multiplier * autocorr_multiplier * sharpe_ratio


# endregion


# region p-value


def sharpe_ratio_t_statistic(
    sharpe_ratio: np.float64, num_observations: int
) -> np.float64:
    """
    Calculate the t-statistic, as a result of
    transforming a Sharpe ratio

    Parameters
    ----------
    sharpe_ratio: np.float64
        The Sharpe ratio
    num_observations: int
        The number of observations present in
        the returns series,
        for which the SR was calculated/adjusted

    Returns
    -------
    np.float64
    The t-statistic of the Sharpe ratio
    """
    t_statistic = sharpe_ratio * np.sqrt(num_observations)

    return t_statistic


def sharpe_ratio_p_value(sharpe_ratio: np.float64, num_observations: int) -> np.float64:
    """
    Calculate the one-tailed p-value of a Sharpe ratio

    First, we obtain the t-statistic of the Sharpe ratio

    Second, the one-tailed p-value is calculated,
    quantifying the statistical significance of the SR

    Parameters
    ----------
    sharpe_ratio: np.float64
        The Sharpe ratio
    num_observations: int
        The number of observations present in
        the returns series,
        for which the SR was calculated/adjusted

    Returns
    -------
    np.float64
    The p-value of the Sharpe ratio
    """
    t_statistic = sharpe_ratio_t_statistic(sharpe_ratio, num_observations)

    degrees_of_freedom = num_observations - 1
    p_value = 1 - stats.t.cdf(t_statistic, df=degrees_of_freedom)

    return p_value


# endregion

# region returns


def equal_weighted_avg_corr(
    returns: pd.DataFrame,
) -> np.float64:
    """
    Calculate the equal weighted average correlation
    between returns

    Note that this approach may result in the
    estimate of average
    itself being overfit if short sample lengths are used.
    Nonetheless, more complex methods are both
    out of the scope
    of the work, as well as require access to HPC

    The summation is implemented in such a way to skip
    rows with all zeroes, since theirs correlation would
    be NaN

    Parameters
    ----------
    returns : pd.DataFrame
        A dataframe, where each row is
        a series of non-annualized returns of the strategy

    Returns
    -------
    np.float64
    The equal weighted average correlation between returns
    """
    num_strategies = returns.shape[0]
    corr_sum = np.float64(0)
    num_pairs = 0

    for i in range(num_strategies):
        if np.count_nonzero(returns.iloc[i]) == 0:
            continue

        for j in range(i + 1, num_strategies):
            if np.count_nonzero(returns.iloc[j]) == 0:
                continue

            corr_sum += returns.iloc[i].corr(returns.iloc[j])
            num_pairs += 1

    # if no rows were skipped, `num_pairs` would be
    # equal to `num_strategies * (num_strategies - 1)`
    # thus making this formula aligned with the one
    # presented in de Prado's work
    average_correlation = 2 * corr_sum / num_pairs

    return average_correlation


# endregion
