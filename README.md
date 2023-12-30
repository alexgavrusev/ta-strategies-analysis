# ta-strategies-analysis

The software part of my bachelor's thesis, [Examining technical analysis-based strategies in Forex and Crypto markets](https://is.vsfs.cz/th/uvjf8/examining-technical-analysis-based-strategies-in-forex-and-crypto-markets.pdf)

## Setup

Create and active a virtual environment

```console
$ python3 -m venv .venv
$ source .venv/bin/activate
```

Install the dependencies

```console
$ python3 -m pip install -r requirements.txt
```

## Used data

The OHLC series are located in `./data`, and can be re-downloaded by using `download_data.py`:

```console
$ python3 src/download_data.py
```

## Getting the results

The first step is to backtest the strategies

```console
$ python3 src/run_strategies.py
```

After that, the returns are analyzed:

```console
$ python3 src/analyze_returns.py
```

Finally, output the "top 10" and "all strategies" LaTeX tables:

```console
$ python3 src/output_tables.py
```

## License

MIT © Aliaksandr Haurusiou.
