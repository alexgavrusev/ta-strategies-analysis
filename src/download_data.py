import pandas as pd
import yfinance as yf

import config as cfg
from utils.output import ensure_empty_dir

if __name__ == "__main__":
    ensure_empty_dir(cfg.data_dir)

    for ticker in cfg.tickers:
        data: pd.DataFrame = yf.download(
            ticker,
            start=cfg.start_date,
            end=cfg.end_date,
            interval=cfg.interval,
        )

        print(f"Downloaded {ticker} from {cfg.start_date} to {cfg.end_date}")

        data.to_csv(cfg.get_ticker_dataname(ticker))
