#!/usr/bin/env python3
# Runs time series foundation models zero-shot on the closes from fetch.py. On every trading day of the
# test window each model sees the last 512 closes of a ticker and forecasts the next 20, with its 10%, 50%
# and 90% quantiles. The same is done for one simulated random walk per ticker (same start price, drift and
# volatility as the real stock), the control: a series that by construction can't be predicted.
#
#   pip install torch pandas chronos-forecasting toto-models git+https://github.com/google-research/timesfm.git
#   python3 forecast.py [model ...]
# Writes data/forecasts/<model>.npz. Needs a CUDA GPU (10 to 60 seconds per model on an RTX 4090).
import os, sys, time
import numpy as np
import pandas as pd
import torch

DIR = 'data'
CONTEXT, HORIZON, TEST_START = 512, 20, '2024-11-01'
QUANTILES = [0.1, 0.5, 0.9]


def load_series():
    prices = pd.read_csv(f'{DIR}/prices.csv', index_col='date', parse_dates=True)
    rng = np.random.default_rng(7)
    fakes = {}
    for ticker in prices.columns:
        log_returns = np.diff(np.log(prices[ticker].values))
        steps = rng.normal(log_returns.mean(), log_returns.std(), len(log_returns))
        fakes[f'{ticker}*'] = prices[ticker].iloc[0] * np.exp(np.concatenate([[0], np.cumsum(steps)]))
    return pd.concat([prices, pd.DataFrame(fakes, index=prices.index)], axis=1)


def timesfm3():
    from timesfm3 import TimesFM3Evaluator, ModelConfig
    model = TimesFM3Evaluator(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch', per_core_batch_size=256, device='cuda'))

    def run(contexts):
        out = model.predict_batch(list(contexts), horizon=HORIZON, return_quantiles=True)
        # Deciles 0.1 to 0.9: indexes 0, 4 and 8 are the 10%, 50% and 90% quantiles.
        return np.stack([o.quantiles[:, [0, 4, 8]] for o in out])
    return run


def chronos2():
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cuda')

    def run(contexts):
        quantiles, _ = pipeline.predict_quantiles(torch.tensor(contexts).unsqueeze(1), prediction_length=HORIZON, quantile_levels=QUANTILES)
        return np.stack([q[0].float().cpu().numpy() for q in quantiles])
    return run


def toto2():
    from toto2 import Toto2Model
    model = Toto2Model.from_pretrained('Datadog/Toto-2.0-313m').to('cuda').eval()

    def run(contexts):
        target = torch.tensor(contexts, device='cuda').unsqueeze(1)
        quantiles = model.forecast({'target': target, 'target_mask': torch.ones_like(target, dtype=torch.bool),
                                    'series_ids': torch.zeros(len(contexts), 1, dtype=torch.long, device='cuda')},
                                   horizon=HORIZON, decode_block_size=768, has_missing_values=False)
        # (9 deciles, batch, 1, horizon) -> (batch, horizon, 3)
        return quantiles[[0, 4, 8], :, 0, :].permute(1, 2, 0).float().cpu().numpy()
    return run


MODELS = {'timesfm-3': timesfm3, 'chronos-2': chronos2, 'toto-2': toto2}

if __name__ == '__main__':
    series = load_series()
    origins = np.where(series.index >= TEST_START)[0]
    origins = origins[origins < len(series) - 1]
    os.makedirs(f'{DIR}/forecasts', exist_ok=True)
    for name in sys.argv[1:] or MODELS:
        run, started = MODELS[name](), time.time()
        out = np.zeros((series.shape[1], len(origins), HORIZON, 3), dtype=np.float32)
        for s, column in enumerate(series.columns):
            values = series[column].values.astype(np.float32)
            # The window ends at the origin's close: the model knows that day's close and forecasts the next ones.
            contexts = np.stack([values[o - CONTEXT + 1:o + 1] for o in origins])
            for b in range(0, len(contexts), 256):
                out[s, b:b + 256] = run(contexts[b:b + 256])
        np.savez_compressed(f'{DIR}/forecasts/{name}.npz', forecasts=out, columns=np.array(series.columns, dtype=str),
                            origins=np.array(series.index[origins].strftime('%Y-%m-%d'), dtype=str))
        print(name, out.shape, f'{time.time() - started:.0f}s')
