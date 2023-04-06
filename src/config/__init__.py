import os
from itertools import product
from pathlib import Path

import backtrader as bt

from .backtest_config import BacktestConfig
from .strategy_class import StrategyClass

from strategies.oscillator_value import (
    OscillatorValueStrategy,
)
from strategies.ma_price_crossover import (
    MAPriceCrossoverStrategy,
)
from strategies.two_ma_crossover import (
    TwoMACrossoverStrategy,
)
from strategies.bb_trend_following import (
    BBTrendFollowingStrategy,
)
from strategies.bb_volatility_breakout import (
    BBVolatilityBreakoutStrategy,
)

from strategies.indicators.mfi import MFI

# region backtrader

broker_cash = 1_000_000.0

# endregion

# region fs

project_root = Path(os.path.abspath(__file__)).parent.parent.parent

data_dir = project_root / "data"

calculations_dir = project_root / "results"
returns_file = calculations_dir / "returns.csv"
raw_results_file = calculations_dir / "raw_results.csv"
all_table_file = calculations_dir / "all.tex"
top_10_table_file = calculations_dir / "top_10.tex"

# endregion


# region data

start_date = "2018-01-01"
end_date = "2023-01-01"
interval = "1d"

crypto_tickers = ["BTC-USD", "ETH-USD"]
fx_tickers = ["EURUSD=X", "JPY=X"]
tickers = crypto_tickers + fx_tickers


def get_ticker_dataname(ticker: str) -> Path:
    return data_dir / f"{ticker}.csv"


# endregion

# region strategies


oscillatior_backtest_configs = [
    BacktestConfig(
        strategy=OscillatorValueStrategy,
        strategy_class=StrategyClass.OSCILLATORS,
        ticker=ticker,
        params={
            "ind": indicator,
            "p": period,
            "os": oversold,
            "ob": overbought,
        },
    )
    for (
        ticker,
        (indicator, oversold, overbought),
        period,
    ) in product(
        tickers,
        [
            (bt.indicators.RSI, 30, 70),
            (bt.indicators.StochasticFast, 80, 20),
            (MFI, 80, 20),
        ],
        [14, 21],
    )
    # FX has no volume info, so MFI wouldn't work there
    if (ticker not in fx_tickers if indicator == MFI else True)
]

ma_price_crossover_backtest_configs = [
    BacktestConfig(
        strategy=MAPriceCrossoverStrategy,
        strategy_class=StrategyClass.MOVING_AVERAGES,
        ticker=ticker,
        params={
            "ma_ind": ma_indicator,
            "p": period,
        },
    )
    for (ticker, ma_indicator, period) in product(
        tickers,
        [bt.indicators.SMA, bt.indicators.EMA],
        [5, 10, 15],
    )
]

two_ma_crossover_backtest_configs = [
    BacktestConfig(
        strategy=TwoMACrossoverStrategy,
        strategy_class=StrategyClass.MOVING_AVERAGES,
        ticker=ticker,
        params={
            "ma_ind": ma_indicator,
            "f_p": fast_period,
            "s_p": slow_period,
        },
    )
    for (
        ticker,
        ma_indicator,
        (fast_period, slow_period),
    ) in product(
        tickers,
        [bt.indicators.SMA, bt.indicators.EMA],
        [(5, 20), (10, 30)],
    )
]

bb_trend_following_backtest_configs = [
    BacktestConfig(
        strategy=BBTrendFollowingStrategy,
        strategy_class=StrategyClass.BOLLINGER_BANDS,
        ticker=ticker,
        params={
            "p": period,
            "df": devfactor,
            "ind": indicator,
        },
    )
    for (
        (ticker, indicator),
        (period, devfactor),
    ) in product(
        [
            # Use RSI only on FX, as that has no volume info
            *product(fx_tickers, [bt.indicators.RSI]),
            # Use MFI only on crypto
            *product(crypto_tickers, [MFI]),
        ],
        [(20, 2.0), (10, 1.9), (50, 2.1)],
    )
]

bb_volatility_backtest_configs = [
    BacktestConfig(
        strategy=BBVolatilityBreakoutStrategy,
        strategy_class=StrategyClass.BOLLINGER_BANDS,
        ticker=crypto_tickers[0],
        params={
            "p": period,
            "df": devfactor,
            "lb_p": lookback_period,
        },
    )
    for ((period, devfactor), lookback_period) in product(
        [(20, 2.0), (10, 1.9), (50, 2.1)], [180, 120]
    )
]

backtest_configs = (
    oscillatior_backtest_configs
    + ma_price_crossover_backtest_configs
    + two_ma_crossover_backtest_configs
    + bb_trend_following_backtest_configs
    + bb_volatility_backtest_configs
)

# endregion

# region analysis

significance_level = 0.05

hlz_num_simulations = 1000

# endregion
