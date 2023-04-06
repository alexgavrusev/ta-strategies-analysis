import multiprocessing

import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd

from config import (
    BacktestConfig,
    backtest_configs,
    broker_cash,
    calculations_dir,
    get_ticker_dataname,
    returns_file,
)
from utils.output import ensure_empty_dir, get_strategy_name


def setup_broker(cerebro: bt.Cerebro):
    cerebro.broker.setcash(broker_cash)


def setup_sizer(cerebro: bt.Cerebro):
    cerebro.addsizer(bt.sizers.PercentSizer, percents=5)


def add_ticker_data(cerebro: bt.Cerebro, ticker: str):
    data = btfeeds.YahooFinanceCSVData(
        dataname=get_ticker_dataname(ticker),
        # rounding at least causes EURUSD to break the fast stoch
        round=False,
    )

    cerebro.adddata(data)


def add_analyzers(cerebro: bt.Cerebro):
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        # use weekly sampling to align
        # forex and crypto returns,
        # and simplify SR annualization
        # thought it is important to note that
        # this reduces the track record length
        timeframe=bt.TimeFrame.Weeks,
        compression=1,
        _name="returns",
    )


def run_strategy(cerebro: bt.Cerebro):
    results = cerebro.run()

    returns = results[0].analyzers.returns.get_analysis()

    strategy = cerebro.runstrats[0][0]
    row_name = f"{get_strategy_name(strategy)}"

    return pd.DataFrame({row_name: returns}).transpose()


def run_backtest(cfg: BacktestConfig):
    cerebro = bt.Cerebro()

    setup_broker(cerebro)
    setup_sizer(cerebro)

    add_ticker_data(cerebro, cfg.ticker)

    cerebro.addstrategy(cfg.strategy, **cfg.params)

    add_analyzers(cerebro)

    df = run_strategy(cerebro)

    # Group all returns into a "Returns" column group
    df.columns = pd.MultiIndex.from_product([["Returns"], df.columns])

    df[("Meta", "Strategy_class")] = cfg.strategy_class.value

    return df


if __name__ == "__main__":
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)

    df_returns_list = pool.map(run_backtest, backtest_configs)

    # create a single dataframe for the returns of all strategies
    df_all_returns = pd.concat(df_returns_list)

    ensure_empty_dir(calculations_dir)
    df_all_returns.to_csv(returns_file)
