import numpy as np
import pandas as pd
from scipy import stats

from .common import equal_weighted_avg_corr


# region Probabilistic
def estimated_sharpe_ratio_stddev(
    returns: pd.Series, sharpe_ratio: np.float64
) -> np.float64:
    """
    Calculate the estimated Sharpe ratio standard deviation

    .. math::
        \\hat{\\sigma}(\\widehat{\\text{SR}}) =
        \\sqrt{
          \\frac{
            1 -
            \\hat{\\gamma}_3
            \\widehat{
              \\text{SR}} +
            \\frac{
              \\hat{\\gamma}_4 - 1}{4}
            \\widehat{
              \\text{SR}}^2
          }{T - 1}
        }

    Parameters
    ----------
    returns : pd.Series
        A series of returns of the strategy, in the
        original sampling frequency
    sharpe_ratio: np.float64
        The non-annualized Sharpe ratio

    Returns
    -------
    np.float64
    The estimated Sharpe ratio standard deviation,
    as derived based on the above-mentioned formula
    """

    skew = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)

    sr_stddev = np.sqrt(
        (1 - skew * sharpe_ratio + (kurtosis - 1) / 4 * sharpe_ratio**2)
        / (returns.size - 1)
    )

    return sr_stddev


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    sr: np.float64,
    sr_benchmark: np.float64,
) -> np.float64:
    """
    Calculate the Probabilistic Sharpe ratio (PSR)

    PSR is a skill metric that states the probability
    of observing a future Sharpe ratio that will be above
    the benchmark.

    2 sources of SR "inflation" are taken into account:
    - The non-normality of returns
    - Infrequent returns sampling

    .. math::
        \\widehat{
          \\text{PSR}}(
        \\text{SR}^*) = \\left[
          \\frac{
            \\widehat{
              \\text{SR}} -
            \\text{SR}^*} {
            \\hat{\\sigma}(\\widehat{\\text{SR}}) } \\right]

    Parameters
    ----------
    returns : pd.Series
        A series of returns of the strategy, in the
        original sampling frequency
    sr: np.float64
        The non-annualized Sharpe ratio
    sr_benchmark: np.float64
        The benchmark Sharpe ratio, usually set to 0

    Returns
    -------
    np.float64
    The Probabilistic Sharpe ratio, with the value
    ranging from 0 to 1
    """

    sr_stddev = estimated_sharpe_ratio_stddev(returns, sr)

    # all-zeroes series
    if np.isnan(sr_stddev):
        return np.float64(0)

    psr = stats.norm.cdf((sr - sr_benchmark) / sr_stddev)

    return psr


# endregion


# region Deflated


def est_num_of_independent_trials(
    returns: pd.DataFrame,
) -> int:
    """
    Calculate the estimated number of independent trials

    Parameters
    ----------
    returns : pd.DataFrame
        A dataframe, where each row is
        a series of non-annualized returns of the strategy

    Returns
    -------
    np.int
    The estimated number of independent trials,
    as defined by the interpolation
    between the extreme cases of 0 and 1 correlation
    """

    num_strategies = returns.shape[0]
    corr = equal_weighted_avg_corr(returns)

    # Ceil up to get an at least slightly lower DSR,
    # minimally compensating for the overfit correlation
    return np.ceil(corr + (1 - corr) * num_strategies)


def estimated_expected_maximum_sharpe_ratio(
    returns: pd.DataFrame, sharpe_ratios: pd.Series
) -> np.float64:
    """
    Calculate the estimated expected maximum sharpe ratio

    Parameters
    ----------
    returns : pd.DataFrame
        A dataframe, where each row is
        a series of non-annualized returns of the strategy
    sharpe_ratios : pd.Series
        A series of non-annualized
        sharpe ratios of the strategies

    Returns
    -------
    np.float64
    The estimated expected maximum sharpe ratio
    """

    variance = sharpe_ratios.var()
    num_trials = est_num_of_independent_trials(returns)

    return np.sqrt(variance) * (
        (1 - np.euler_gamma) * stats.norm.ppf(1 - (1 / num_trials))
        + np.euler_gamma * stats.norm.ppf(1 - (1 / num_trials) * np.e**-1)
    )


def deflated_sharpe_ratio(
    returns: pd.Series,
    sr: np.float64,
    expected_max_sr: np.float64,
) -> np.float64:
    """
    Calculate the Deflated Sharpe ratio (DSR)

    DSR is PSR with
    the benchmark Sharpe ratio calculated in a way
    to account for multiple testing

    .. math::
        \\widehat{
          \text{DSR}} \\equiv
        \\widehat{
          \\text{PSR}}(\\widehat{
          \\text{SR}}_0)


        \\widehat{
          \\text{SR}}_0 =
        \\sqrt{
          V[\\{
              \\widehat{
                \\text{SR}_n}\\}]
        }
        \\left(
        \\left(1 - \\gamma\\right)Z^{-1}
        \\left[
          1 -
          \\frac{1}{N}
          \\right] + \\gamma Z^{-1}\\left[
          1 -
          \\frac{1}{N}e^{-1}
          \\right]\\right)

    Parameters
    ----------
    returns : pd.Series
        A series of returns of the strategy, in the
        original sampling frequency
    sr: np.float64
        The non-annualized Sharpe ratio
    expected_max_sr: np.float64
        The estimated expected maximum sharpe ratio

    Returns
    -------
    np.float64
    The Deflated Sharpe ratio, with the value
    ranging from 0 to 1
    """

    return probabilistic_sharpe_ratio(returns, sr, sr_benchmark=expected_max_sr)


# endregion
