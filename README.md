# time-series-modelling
This repository contains an implementation of the model described in the first two parts of "Active Portfolio Management" by Grinold and Kahn. It exists because I want to teach myself about quantitative modelling and time series analysis. The goal is to implement the risk model using freely available stock price data from `yfinance` and produce a back-tested portfolio whose performance can be compared against a benchmark. The idea is to understand how the model works in theory and in practice, not to actually find new alpha and achieve maximum performance. 

## Overview

The core logic is implemented in a single file (`core.py`) which contains some sanity-check tests. These need to be made into actual tests and moved into their own file. A more detailed
mathematical description of what exactly is being implemented is provided in `notes.pdf`. This is also incomplete.

## Project Structure

- `core.py` – core algorithms and computations
- `SP500.csv` - list of S&P 500 tickers from [here](https://gist.github.com/ZeccaLehn/f6a2613b24c393821f81c0c1d23d4192)
- `market_data.pkl`,`securities_data.pkl` - saved copies of `yfinance` data to avoid redownloading every time
- `X.pkl`, `f.pkl`,`F_mat.pkl`,`D.pkl` - saved copies of $X, f, F, D$ timeseries to avoid recomputing every time the portfolio problem is formulated and solved
- `notes.pdf` – mathematical descriptions of the model and rationale for implementation decisions

## Status

This project is under active development. 


## Usage

The script (`core.py`) runs the entire pipeline when executed.

To run:
```bash
python core.py 
```

## Notes

The emphasis is on correctness and clarity (to support my conceptual understanding) rather than performance, although it runs just fine on an m1 macbook.
