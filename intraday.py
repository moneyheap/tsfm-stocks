#!/usr/bin/env python3
# The same zero-shot test on intraday bars: hourly (Yahoo keeps the last 730 days) and 5-minute (the last 60
# days), regular trading hours only. Each model sees the last 512 bars and forecasts the next 12; we score the
# next bar and the 12th bar (12 hours of trading for hourly bars, one hour for 5-minute bars), on the real
# tickers and on random-walk twins built from each ticker's own bar returns.
#
#   python3 intraday.py [1h|5m ...]
# Writes data/intraday-<interval>.csv (the bars) and intraday-results.json.
import json, os, sys, time
import numpy as np
import pandas as pd
import yfinance as yf
import forecast
from forecast import DIR, CONTEXT, MODELS
from analyze import day_ci

TICKERS = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'AMD',
           'NFLX', 'JPM', 'V', 'XOM', 'UNH', 'JNJ', 'PG', 'KO', 'WMT', 'HD', 'COST', 'CAT']
PERIOD = {'1h': '730d', '5m': '60d'}
HORIZON, STEPS = 12, [1, 12]
forecast.HORIZON = HORIZON  # the model wrappers read it when they run


def bars(interval):
    path = f'{DIR}/intraday-{interval}.csv'
    if not os.path.exists(path):
        raw = yf.download(TICKERS, period=PERIOD[interval], interval=interval, auto_adjust=True, prepost=False, progress=False)
        raw['Close'][TICKERS].dropna().to_csv(path)
    real = pd.read_csv(path, index_col=0)
    real.index = pd.to_datetime(real.index, utc=True).tz_convert('America/New_York')
    # Full sessions only: drops today's unfinished bars, half days and days with a missing bar.
    per_day = pd.Series(real.index.date).map(pd.Series(real.index.date).value_counts())
    real = real[(per_day == per_day.max()).values]
    rng = np.random.default_rng(7)
    fakes = {}
    for t in real.columns:
        r = np.diff(np.log(real[t].values))
        fakes[f'{t}*'] = real[t].iloc[0] * np.exp(np.concatenate([[0], np.cumsum(rng.normal(r.mean(), r.std(), len(r)))]))
    return pd.concat([real, pd.DataFrame(fakes, index=real.index)], axis=1)


def evaluate(series, out, origins):
    values = series.values.T  # (series, bars)
    real = np.array([not c.endswith('*') for c in series.columns])
    days = series.index[origins].normalize()
    result = {}
    for h in STEPS:
        ok = origins + h < values.shape[1]
        base, actual = values[:, origins[ok]], values[:, origins[ok] + h]
        pred, move = out[:, ok, h - 1] / base - 1, actual / base - 1
        hits = np.where((pred == 0) | (move == 0), np.nan, (np.sign(pred) == np.sign(move)).astype(float))
        # One observation per trading day for the margin, as in analyze.py: bars on the same day move together.
        daily = pd.DataFrame(hits.T).groupby(np.asarray(days[ok])).mean().values.T
        row = {}
        for group, rows in (('real', real), ('randomWalk', ~real)):
            rate = float(np.nanmean(hits[rows]))
            row[group] = {'hitRate': rate, 'hitRateCi': day_ci(daily[rows])[1],
                          'alwaysUpRate': float(np.mean(move[rows][move[rows] != 0] > 0)),
                          'errorVsNoChange': float(np.mean(np.abs(pred[rows] - move[rows])) / np.mean(np.abs(move[rows]))),
                          'correlation': float(np.corrcoef(pred[rows].ravel(), move[rows].ravel())[0, 1]),
                          'medianAbsPredicted': float(np.median(np.abs(pred[rows]))), 'medianAbsReal': float(np.median(np.abs(move[rows])))}
        result[h] = row
    return result


if __name__ == '__main__':
    path = f'{DIR}/intraday-results.json'
    results = json.load(open(path)) if os.path.exists(path) else {}
    for interval in sys.argv[1:] or PERIOD:
        series = bars(interval)
        origins = np.arange(CONTEXT - 1, len(series) - 1)
        results[interval] = {'bars': len(series), 'from': str(series.index[0]), 'to': str(series.index[-1]), 'origins': len(origins), 'models': {}}
        for name in MODELS:
            run, started = MODELS[name](), time.time()
            out = np.zeros((series.shape[1], len(origins), HORIZON), dtype=np.float32)
            for s, column in enumerate(series.columns):
                values = series[column].values.astype(np.float32)
                contexts = np.stack([values[o - CONTEXT + 1:o + 1] for o in origins])
                for b in range(0, len(contexts), 256):
                    out[s, b:b + 256] = run(contexts[b:b + 256])[..., 1]
            results[interval]['models'][name] = evaluate(series, out, origins)
            print(interval, name, f'{time.time() - started:.0f}s', json.dumps({h: {g: round(v['hitRate'], 4) for g, v in r.items()} for h, r in results[interval]['models'][name].items()}))
        json.dump(results, open(path, 'w'), indent=2)
