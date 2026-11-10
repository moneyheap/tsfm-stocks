#!/usr/bin/env python3
# Scores the forecasts from forecast.py against what happened, runs the simple backtest and draws the
# charts. Writes data/results.json and charts/*.svg.
#
#   pip install pandas matplotlib
#   python3 analyze.py
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forecast import DIR, load_series

CHARTS = 'charts'
MODELS = {'timesfm-3': 'TimesFM 3.0 (Google)', 'chronos-2': 'Chronos-2 (Amazon)', 'toto-2': 'Toto 2.0 (Datadog)'}
COLORS = {'timesfm-3': '#2a78d6', 'chronos-2': '#eb6834', 'toto-2': '#1baf7a'}
INK, INK_2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e7e6e2', '#fcfcfb'
HORIZONS = [1, 5, 20]
COST = 0.0005  # 0.05% each time a position is opened or closed
EXAMPLE_ORIGIN = '2025-03-28'  # the Friday before the April 2025 tariff sell-off

plt.rcParams.update({
    'svg.fonttype': 'none', 'font.family': 'sans-serif', 'font.sans-serif': ['Inter', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.edgecolor': GRID, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'legend.frameon': False, 'lines.linewidth': 2,
})


def score(forecasts, columns, origin_rows, prices):
    """Per series, origin and horizon: forecast return, real return, and the 10-90% range in price terms."""
    values = prices[columns].values  # (days, series)
    last = values[origin_rows].T  # (series, origins)
    out = {}
    for h in HORIZONS:
        target_rows = origin_rows + h
        valid = target_rows < len(values)
        actual = values[target_rows[valid]].T
        f = forecasts[:, valid, h - 1, :]
        base = last[:, valid]
        log_returns = np.diff(np.log(values), axis=0)
        # The volatility rule: a normal 80% range from the last 60 days' volatility, scaled by sqrt(h).
        vol = np.stack([log_returns[r - 60:r].std(axis=0) for r in origin_rows[valid]]).T * np.sqrt(h)
        out[h] = {
            'pred': f[..., 1] / base - 1, 'real': actual / base - 1,
            'inside': (f[..., 0] <= actual) & (actual <= f[..., 2]),
            'inside_vol': np.abs(np.log(actual / base)) <= 1.2816 * vol,
        }
    return out


def day_ci(hits):
    """Mean and 95% range of a (series, days) hit matrix, treating each day as one observation."""
    daily = np.nanmean(hits, axis=0)
    return float(np.nanmean(hits)), float(1.96 * np.nanstd(daily) / np.sqrt(np.sum(~np.isnan(daily))))


def metrics(s, rows):
    out = {}
    for h in HORIZONS:
        pred, real = s[h]['pred'][rows], s[h]['real'][rows]
        hits = np.where((pred == 0) | (real == 0), np.nan, (np.sign(pred) == np.sign(real)).astype(float))
        up = np.where(real == 0, np.nan, (real > 0).astype(float))
        rate, ci = day_ci(hits)
        out[h] = {
            'hitRate': rate, 'hitRateCi': ci, 'alwaysUpRate': float(np.nanmean(up)),
            'predictedUpShare': float(np.mean(pred > 0)),
            'errorVsNoChange': float(np.mean(np.abs(pred - real)) / np.mean(np.abs(real))),
            'correlation': float(np.corrcoef(pred.ravel(), real.ravel())[0, 1]),
            'medianAbsPredicted': float(np.median(np.abs(pred))), 'medianAbsReal': float(np.median(np.abs(real))),
            'coverage80': float(np.mean(s[h]['inside'][rows])), 'coverage80VolRule': float(np.mean(s[h]['inside_vol'][rows])),
        }
    return out


def backtest(s, rows, cost=COST):
    """Hold each stock on the days its model forecasts a higher close tomorrow, cash otherwise; equal weight."""
    pred, real = s[1]['pred'][rows], s[1]['real'][rows]
    position = (pred > 0).astype(float)
    trades = np.abs(np.diff(np.concatenate([np.zeros((len(position), 1)), position], axis=1), axis=1))
    daily = (position * real - cost * trades).mean(axis=0)
    return daily, float(position.mean()), float(trades.sum(axis=1).mean())


def summary(daily):
    growth = np.cumprod(1 + daily)
    years = len(daily) / 252
    return {
        'totalReturn': float(growth[-1] - 1), 'annualReturn': float(growth[-1] ** (1 / years) - 1),
        'annualVolatility': float(daily.std() * np.sqrt(252)),
        'sharpe': float(daily.mean() / daily.std() * np.sqrt(252)),
        'maxDrawdown': float((growth / np.maximum.accumulate(growth) - 1).min()),
    }


def save(fig, name):
    fig.savefig(f'{CHARTS}/{name}.svg', bbox_inches='tight', pad_inches=0.25, dpi=200, metadata={'Date': None})
    plt.close(fig)


def pct(axis, decimals=0):
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v * 100:.{decimals}f}%'))


if __name__ == '__main__':
    os.makedirs(CHARTS, exist_ok=True)
    prices = load_series()
    real_cols = [c for c in prices.columns if not c.endswith('*')]
    loaded = {m: np.load(f'{DIR}/forecasts/{m}.npz') for m in MODELS}
    first = next(iter(loaded.values()))
    columns, origins = first['columns'].tolist(), first['origins'].tolist()
    origin_rows = np.array([prices.index.get_loc(pd.Timestamp(o)) for o in origins])
    real_rows = np.array([c in real_cols for c in columns])
    scored = {m: score(loaded[m]['forecasts'], columns, origin_rows, prices) for m in MODELS}

    results = {'testWindow': [origins[0], str(prices.index[-1].date())], 'origins': len(origins),
               'tickers': real_cols, 'models': {}}
    for m in MODELS:
        results['models'][m] = {'real': metrics(scored[m], real_rows), 'randomWalk': metrics(scored[m], ~real_rows)}
        per_ticker = {}
        for i, c in enumerate(columns):
            if c in real_cols:
                pred, real = scored[m][1]['pred'][i], scored[m][1]['real'][i]
                per_ticker[c] = float(np.mean(np.sign(pred) == np.sign(real)))
        results['models'][m]['hitRateByTicker'] = per_ticker

    # Backtest on the real stocks.
    hold_daily = scored['timesfm-3'][1]['real'][real_rows].mean(axis=0)
    curves = {'hold': hold_daily}
    results['backtest'] = {'costPerTrade': COST, 'buyAndHold': summary(hold_daily)}
    real = scored['timesfm-3'][1]['real'][real_rows]
    rng = np.random.default_rng(11)
    for m in MODELS:
        daily, invested, trades = backtest(scored[m], real_rows)
        curves[m] = daily
        free = summary(backtest(scored[m], real_rows, cost=0)[0])
        # The luck check: 2,000 coin-flip timings in the market on the same share of days, before costs.
        coins = [summary((rng.random(real.shape) < invested).astype(float).__mul__(real).mean(axis=0))['sharpe'] for _ in range(2000)]
        results['backtest'][m] = {**summary(daily), 'shareOfDaysInvested': invested, 'tradesPerStock': trades,
                                  'withoutCosts': {**free, 'coinSharpeBeaten': float(np.mean(np.array(coins) < free['sharpe'])),
                                                   'coinSharpeMedian': float(np.median(coins))}}
    json.dump(results, open(f'{DIR}/results.json', 'w'), indent=2)

    # Chart 1: one example forecast, SPY before the April 2025 sell-off.
    spy = columns.index('SPY')
    o = origins.index(EXAMPLE_ORIGIN)
    row = origin_rows[o]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    hist = prices['SPY'].iloc[row - 80:row + 1]
    after = prices['SPY'].iloc[row:row + 21]
    ax.plot(hist.index, hist.values, color=INK, linewidth=1.6, label='SPY close')
    ax.plot(after.index, after.values, color=INK, linewidth=1.6, linestyle=(0, (2, 2)), label='What happened next')
    future = prices.index[row:row + 21]
    for m in MODELS:
        f = loaded[m]['forecasts'][spy, o]
        median = np.concatenate([[hist.values[-1]], f[:, 1]])
        ax.plot(future, median, color=COLORS[m], label=f'{MODELS[m]} forecast')
        if m == 'timesfm-3':
            ax.fill_between(future[1:], f[:, 0], f[:, 2], color=COLORS[m], alpha=0.14, linewidth=0, label='TimesFM 10-90% range')
    ax.axvline(prices.index[row], color=MUTED, linewidth=1)
    ax.text(prices.index[row], ax.get_ylim()[1], ' forecast made', color=INK_2, fontsize=9, va='top')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'${v:.0f}'))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d %b'))
    ax.set_title('SPY on 28 March 2025: what three models forecast, and what happened', loc='left', fontsize=12, color=INK)
    ax.legend(loc='lower left', fontsize=9, ncol=2)
    save(fig, 'spy-april-2025')

    # Chart 2: direction hit rate (next day), real stocks vs random walks, with day-level 95% ranges.
    fig, ax = plt.subplots(figsize=(8, 3.6))
    up = results['models']['timesfm-3']['real'][1]['alwaysUpRate']
    ax.axvline(0.5, color=MUTED, linewidth=1)
    ax.axvline(up, color=INK_2, linewidth=1, linestyle=(0, (3, 2)))
    ax.text(0.5, 3.45, ' coin flip', color=INK_2, fontsize=9, va='center')
    ax.text(up, 3.45, f' "always up" ({up * 100:.1f}%)', color=INK_2, fontsize=9, va='center')
    for i, m in enumerate(MODELS):
        for group, offset, fill in (('real', 0.14, True), ('randomWalk', -0.14, False)):
            r = results['models'][m][group][1]
            y = 2 - i + offset
            ax.errorbar(r['hitRate'], y, xerr=r['hitRateCi'], color=COLORS[m], capsize=0, linewidth=2, zorder=2)
            ax.plot(r['hitRate'], y, 'o', markersize=8, color=COLORS[m] if fill else SURFACE, markeredgecolor=COLORS[m], markeredgewidth=2, zorder=3)
            ax.text(0.553, y, f"{r['hitRate'] * 100:.1f}%", fontsize=9, color=INK_2, va='center')
    ax.set_yticks([2, 1, 0], [MODELS[m] for m in MODELS])
    ax.set_ylim(-0.5, 3.7)
    ax.set_xlim(0.47, 0.56)
    pct(ax.xaxis)
    ax.grid(axis='y', visible=False)
    ax.plot([], [], 'o', color=INK_2, label='24 real stocks and ETFs')
    ax.plot([], [], 'o', color=SURFACE, markeredgecolor=INK_2, markeredgewidth=2, label='24 simulated random walks')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=9, ncol=2)
    ax.set_xlabel('Share of days the model got the direction of the next close right')
    ax.set_title('Next-day direction: about as good as a coin, on real and fake prices', loc='left', fontsize=12)
    save(fig, 'direction-hit-rate')

    # Chart 3: forecast vs actual next-day move, TimesFM 3.0, same scale on both axes.
    fig, ax = plt.subplots(figsize=(6, 6))
    pred, real = scored['timesfm-3'][1]['pred'][real_rows].ravel(), scored['timesfm-3'][1]['real'][real_rows].ravel()
    ax.scatter(real, pred, s=4, color=COLORS['timesfm-3'], alpha=0.25, linewidths=0, rasterized=True)
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.plot([-0.1, 0.1], [-0.1, 0.1], color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
    ax.text(0.05, 0.058, 'perfect forecast', color=INK_2, fontsize=9, rotation=45, ha='center', va='center')
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
        pct(axis)
    ax.set_xlabel('Actual next-day move')
    ax.set_ylabel('TimesFM 3.0 forecast of the move')
    ax.set_title('Forecast vs actual next-day move, 24 tickers', loc='left', fontsize=12)
    save(fig, 'forecast-vs-actual')

    # Chart 4: the backtest, growth of $10,000.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    dates = pd.to_datetime(origins)
    for key, label, color in [('hold', 'Hold all 24, every day', INK)] + [(m, MODELS[m], COLORS[m]) for m in MODELS]:
        growth = 10000 * np.cumprod(1 + curves[key])
        ax.plot(dates[:len(growth)], growth, color=color, linewidth=2 if key == 'hold' else 1.8, label=label)
        ax.annotate(f'${growth[-1]:,.0f}', (dates[len(growth) - 1], growth[-1]), xytext=(6, 0), textcoords='offset points',
                    fontsize=9, color=INK_2, va='center')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %Y'))
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('$10,000, holding each stock only on days its model forecast a rise', loc='left', fontsize=12)
    save(fig, 'backtest')

    # Chart 5: how often the real price landed inside the model's 80% range.
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.axvline(0.8, color=INK_2, linewidth=1, linestyle=(0, (3, 2)))
    ax.text(0.8, 3.45, ' 80% (well calibrated)', color=INK_2, fontsize=9, va='center')
    rows = [(MODELS[m], [results['models'][m]['real'][h]['coverage80'] for h in HORIZONS], COLORS[m]) for m in MODELS]
    rows.append(('Volatility rule (no AI)', [results['models']['timesfm-3']['real'][h]['coverage80VolRule'] for h in HORIZONS], INK_2))
    markers = {1: 'o', 5: 's', 20: 'D'}
    for i, (label, values, color) in enumerate(rows):
        y = 3 - i
        for h, v in zip(HORIZONS, values):
            ax.plot(v, y, markers[h], markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5)
    for h in HORIZONS:
        ax.plot([], [], markers[h], color=INK_2, label=f'{h} day' + ('s' if h > 1 else '') + ' ahead')
    ax.set_yticks([3, 2, 1, 0], [r[0] for r in rows])
    ax.set_ylim(-0.5, 3.8)
    pct(ax.xaxis)
    ax.grid(axis='y', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), fontsize=9, ncol=3)
    ax.set_xlabel('Share of real outcomes inside the forecast 10-90% range')
    ax.set_title('Forecast ranges: close to 80%, like a simple volatility rule', loc='left', fontsize=12)
    save(fig, 'range-coverage')

    # Chart 6: a real stock and its random-walk twin over the test window, indexed to 100.
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    start = origin_rows[0]
    for ax, (column, title) in zip(axes, [('NVDA', 'Nvidia (real)'), ('NVDA*', 'Random walk built to match Nvidia')]):
        path = prices[column].values[start:] / prices[column].values[start] * 100
        ax.plot(prices.index[start:], path, color=INK if column == 'NVDA' else MUTED, linewidth=1.6)
        ax.axhline(100, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.set_title(title, loc='left', fontsize=11)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %y'))
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 7]))
    save(fig, 'real-or-fake')

    print(json.dumps({m: {g: {h: {k: round(v, 4) for k, v in results['models'][m][g][h].items()} for h in HORIZONS}
                          for g in ('real', 'randomWalk')} for m in MODELS}, indent=1))
    print(json.dumps(results['backtest'], indent=1))
