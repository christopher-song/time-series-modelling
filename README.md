# time-series-modelling
This repository contains an implementation of the model described in the first two parts of "Active Portfolio Management" by Grinold and Kahn. It exists because I want to teach myself about quantitative modelling and time series analysis. The goal is to implement the risk model using freely available stock price data from `yfinance` and produce a back-tested portfolio whose performance can be compared against a benchmark. The idea is to understand how the model works in theory and in practice, not to achieve maximum performance. 

## Overview

The core logic is implemented in a single file (`core.py`) which contains some sanity-check tests. These need to be made into actual tests and moved into their own file. A more detailed
mathematical description of what exactly is being implemented is provided in `notes.pdf`. This is also incomplete.

## Project Structure

- `core.py` – core algorithms and computations
- `notes.pdf` – mathematical descriptions of the model and rationale for implementation decisions

## Status

This project is under active development. 

## Notes

The emphasis correctness and clarity (basically conceptual understanding for me) rather than performance, although it runs just fine on a laptop.
