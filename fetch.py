#!/usr/bin/env python3
# Daily closes from Yahoo Finance for the time series foundation model test. Adjusted for splits and dividends.
#
#   pip install yfinance pandas
#   python3 fetch.py
import os
import yfinance as yf

DIR = 'data'
# Three index ETFs and 21 large US stocks across sectors.
TICKERS = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'AMD', 'NFLX',
           'JPM', 'V', 'XOM', 'UNH', 'JNJ', 'PG', 'KO', 'WMT', 'HD', 'COST', 'CAT']

os.makedirs(DIR, exist_ok=True)
closes = yf.download(TICKERS, start='2014-01-01', end='2026-11-07', auto_adjust=True, progress=False)['Close']
closes = closes[TICKERS].dropna()
closes.index = closes.index.date
closes.to_csv(f'{DIR}/prices.csv', index_label='date', float_format='%.4f')
print(len(closes), 'days,', closes.index[0], 'to', closes.index[-1])
