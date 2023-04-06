import numpy as np
import pandas as pd

from config import (
    hlz_num_simulations,
    raw_results_file,
    returns_file,
)
from stats.common import (
    annualized_estimated_sharpe_ratio,
    equal_weighted_avg_corr,
    estimated_sharpe_ratio,
    sharpe_ratio_p_value,
)
from stats.deflated import (
    deflated_sharpe_ratio,
    estimated_expected_maximum_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from stats.haircut import hlz_p_value


def calculate_base_stats(returns: pd.Series):
    sr = estimated_sharpe_ratio(returns)
    sr_p_value = sharpe_ratio_p_value(sr, returns.size)

    ann_sr = annualized_estimated_sharpe_ratio(returns, 52)  # weeks in a year
    psr = probabilistic_sharpe_ratio(returns, sr, sr_benchmark=np.float64(0))

    return pd.Series(
        {
            "SR": sr,
            "SR_p_value": sr_p_value,
            "Ann_SR": ann_sr,
            "PSR": psr,
        }
    )


if __name__ == "__main__":
    # region base
    all_strategies_info = pd.read_csv(returns_file, index_col=0, header=[0, 1])

    all_strategies_returns = all_strategies_info["Returns"]

    statistics = all_strategies_returns.apply(calculate_base_stats, axis=1)

    statistics["Strategy_class"] = all_strategies_info[("Meta", "Strategy_class")]
    # endregion

    # region DSR

    expected_max_sr = estimated_expected_maximum_sharpe_ratio(
        all_strategies_info["Returns"], statistics["SR"]
    )

    statistics["DSR"] = [
        deflated_sharpe_ratio(
            returns,
            statistics.loc[idx, "SR"],
            expected_max_sr,
        )
        for (
            idx,
            returns,
        ) in all_strategies_returns.iterrows()
    ]
    # endregion

    # region HSR
    returns_corr = equal_weighted_avg_corr(all_strategies_returns)

    num_trials, returns_len = all_strategies_returns.shape

    statistics["HLZ_p_value"] = [
        hlz_p_value(
            ann_sr,
            52,
            returns_len,
            returns_corr,
            num_trials,
            hlz_num_simulations,
        )
        for ann_sr in statistics["Ann_SR"]
    ]
    # endregion

    statistics.to_csv(raw_results_file)
