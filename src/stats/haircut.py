from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg, stats

from .common import sharpe_ratio_p_value

# region p-value adjustments


def bhy_adjustment(p_values_asc: np.ndarray) -> np.ndarray:
    """
    Applies the Benjamini-Hochberg-Yekutieli (BHY)
    multiple testing adjustment to a series of p-values

    Parameters
    ----------
    p_values_asc: np.ndarray
        A series of p-values, sorted in ascending order

    Returns
    -------
    np.ndarray
    The adjusted p-values
    """

    n = p_values_asc.size

    normalizing_constant = np.sum(1 / np.arange(1, n + 1))
    adjusted_p_values = np.zeros(n)

    # iterate backwards from (n - 1) to 0
    for i in np.arange(n - 1, -1, -1):
        # for the last element,
        # the value is simply the p-value
        if i == n - 1:
            adjusted_p_values[i] = p_values_asc[i]
        # for other elements, the value is
        # the min of the adjusted one,
        # or the next element
        else:
            multiplier = (n * normalizing_constant) / (i + 1)

            adjusted_p_values[i] = min(
                adjusted_p_values[i + 1],
                p_values_asc[i] * multiplier,
            )

    return adjusted_p_values


# endregion

# region simulation


@dataclass
class SimulationParameters:
    """
    Dataclass describing the simulation parameters
    for the HLZ model
    """

    rho: np.float64
    total_num_trials: int
    prob_zero_mean: np.float64
    lambd: np.float64


def get_simulation_parameters(
    returns_corr: np.float64, num_trials: int
) -> SimulationParameters:
    """
    Returns the simulation parameters for the HLZ model,
    using linear interpolation, based on the value of the
    returns correlation

    Parameters
    ----------
    returns_corr: np.float64
        Equal-weighted average correlation between
        the returns of all tested strategies

    Returns
    -------
    SimulationParameters
    Simulation parameters series for the HLZ model
    """

    if returns_corr < 0 or returns_corr > 1:
        raise ValueError("Returns correlation should be between 0 and 1")

    parameters_df = pd.DataFrame(
        {
            "rho": [0.0, 0.2, 0.4, 0.6, 0.8],
            "total_num_trials": [
                1295,
                1377,
                1476,
                1773,
                3109,
            ],
            "prob_zero_mean": [
                3.9660 * 0.1,
                4.4589 * 0.1,
                4.8604 * 0.1,
                5.9902 * 0.1,
                8.3901 * 0.1,
            ],
            "lambd": [
                5.4995 * 0.001,
                5.5508 * 0.001,
                5.5413 * 0.001,
                5.5512 * 0.001,
                5.5956 * 0.001,
            ],
        }
    ).set_index("rho", drop=False)

    rho1 = 0.0
    rho2 = 0.2
    if returns_corr < 0.2:
        rho1 = 0.0
        rho2 = 0.2
    elif returns_corr < 0.4:
        rho1 = 0.2
        rho2 = 0.4
    elif returns_corr < 0.6:
        rho1 = 0.4
        rho2 = 0.6
    else:
        rho1 = 0.6
        rho2 = 0.8

    rho1_params = parameters_df.loc[rho1]
    rho2_params = parameters_df.loc[rho2]

    rho1_weight = (rho2 - returns_corr) / (rho2 - rho1)
    rho2_weight = (returns_corr - rho1) / (rho2 - rho1)

    interpolated_parameters = rho1_params * rho1_weight + rho2_params * rho2_weight

    # cast to int, and ensure
    # that is greater than `num_trials`
    total_num_trials = int(
        np.floor((num_trials / interpolated_parameters["total_num_trials"]) + 1)
        * np.floor(interpolated_parameters["total_num_trials"] + 1)
    )

    return SimulationParameters(
        # since series are homogeneous, providing
        # the total_num_trials is provided separately
        # for it to actually be an int, not a float
        **interpolated_parameters.drop("total_num_trials").to_dict(),
        total_num_trials=total_num_trials
    )


def generate_t_stats_panel(
    params: SimulationParameters, num_simulations: int
) -> np.ndarray:
    """
    Generate a panel of t-statistics, as per the HLZ model

    Parameters
    ----------
    params: SimulationParameters
        The simulation parameters
    num_simulations: int
        The number of simulations that will be performed

    Returns
    -------
    np.ndarray
    A 2D array of t-ratios of shape
    (num_simulations, params.total_num_trials)
    """

    # Default parameters of the model
    monthly_volatility = 0.15 / np.sqrt(12)
    num_observations = 240

    correlation_vector = np.insert(
        params.rho * np.ones((1, params.total_num_trials - 1)),
        0,
        1,
    )

    correlation_matrix = linalg.toeplitz(correlation_vector)

    mean = np.zeros(params.total_num_trials)

    covariance_matrix = correlation_matrix * (
        monthly_volatility**2 / num_observations
    )

    # NOTE: slow, takes >1s, impl compatible with numba
    #  would be nice
    shock_matrix = np.random.multivariate_normal(
        mean, covariance_matrix, num_simulations
    )

    prob_vec = np.random.uniform(0, 1, (num_simulations, params.total_num_trials))

    mean_vec = np.random.exponential(
        params.lambd,
        (num_simulations, params.total_num_trials),
    )

    mu_null = np.multiply(prob_vec > params.prob_zero_mean, mean_vec)

    t_stats_panel = abs(mu_null + shock_matrix) / (
        monthly_volatility / np.sqrt(num_observations)
    )

    return t_stats_panel


def t_panel_to_p_panel(t_stats_panel: np.ndarray, num_trials: int) -> np.ndarray:
    """
    Convert a panel of t-ratios into a panel of p-values

    Parameters
    ----------
    t_stats_panel: np.ndarray
        The panel of t-ratios

    Returns
    -------
    np.ndarray
    A 2D array of p-values, with the same shape
    """

    n_sim = t_stats_panel.shape[0]

    p_panel = np.zeros((n_sim, num_trials - 1))

    for i in range(n_sim):
        simulated_t_ratios = t_stats_panel[i, 0 : (num_trials - 1)]
        p_panel[i] = 1 - stats.norm.cdf(simulated_t_ratios, 0, 1)

    return p_panel


# endregion

# region core


def get_multiple_testing_adjusted_p_value(
    p_values_panel: np.ndarray, p_value: np.float64
):
    """
    Get a multiple testing adjusted p-value,
    given a strategy (single-test) p-value

    Parameters
    ----------
    p_values_panel: np.ndarray
        The panel of p-values
    p_value: np.float64
        The p-value of the strategy

    Returns
    -------
    np.float64
        The multiple testing adjusted p-value
    """

    n_sim = p_values_panel.shape[0]

    p_values_bhy = np.ones(n_sim)

    for i in range(0, n_sim):
        simulated_p_values = np.sort(p_values_panel[i])

        # add the p-value of the examined strategy,
        # preserving sorting
        p_val_insert_idx = np.searchsorted(simulated_p_values, p_value)
        p_values = np.insert(simulated_p_values, p_val_insert_idx, p_value)

        # get the adjusted p-values of the examined strategy
        p_bhy: np.float64 = bhy_adjustment(p_values)[p_val_insert_idx]

        p_values_bhy[i] = p_bhy

    p_value_bhy = np.median(p_values_bhy)

    return p_value_bhy


def hlz_p_value(
    ann_sr: np.float64,
    periods: int,
    returns_len: int,
    returns_corr: np.float64,
    num_trials: int,
    num_simulations: int,
) -> np.float64:
    """
    Get the multiple testing adjusted p-value,
    based on the work of Harvey, Liu, and Zhu

    Parameters
    ----------
    ann_sr: np.float64
        The annualized Sharpe ratio
    periods : int
        Frequency of the returns,
        e.g. 52 when weekly returns are provided
    returns_len: int
        The number of observations
        present in the returns series
    returns_corr: np.float64
        Equal-weighted average correlation between
        the returns of all tested strategies
    num_trials: int
        Total number of trials (strategies tested)
    num_simulations: int
        Number of simulations to use to sample the p-values

    Returns
    -------
    np.float64
    The multiple testing adjusted p-value
    """

    num_monthly_observations = np.floor(returns_len * 12 / periods)
    monthly_sr = ann_sr / np.sqrt(12)
    sr_p_value = sharpe_ratio_p_value(monthly_sr, num_monthly_observations)

    simulation_parameters = get_simulation_parameters(returns_corr, num_trials)

    t_stats_panel = generate_t_stats_panel(simulation_parameters, num_simulations)
    p_values_panel = t_panel_to_p_panel(t_stats_panel, num_trials)

    mult_test_p_value = get_multiple_testing_adjusted_p_value(
        p_values_panel, sr_p_value
    )

    return mult_test_p_value


# endregion
