"""Weekly avocado demand forecasting, adapted from an internship project.

Run: python forecast.py
All evaluation forecasts are recursive: future actual quantities are never inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
SIZES = (32, 48, 60)
HORIZON = 4
LAGS = 4
FEATURES = ['week_sin', 'week_cos', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'mean_4']


def monday_start(dates):
    """Monday containing each date (W-MON would instead start on Tuesday)."""
    return dates.dt.normalize() - pd.to_timedelta(dates.dt.weekday, unit='D')


def clean_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    required = {'date', 'company', 'customer_id', 'order_id', 'quantity'}
    if required - set(df):
        raise ValueError(f'Missing columns: {sorted(required - set(df))}')
    audit = {'input_rows': len(df), 'duplicate_rows_removed': int(df.duplicated().sum())}
    df = df.drop_duplicates().copy()
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    valid = df[['date', 'company', 'customer_id', 'quantity']].notna().all(axis=1) & df['quantity'].ge(0)
    audit['invalid_rows_removed'] = int((~valid).sum())
    df = df.loc[valid].copy()
    for key in ['company', 'customer_id', 'order_id']:
        df[key] = df[key].astype('string').str.strip()
    product = df.get('product', pd.Series('', index=df.index)).astype('string').str.lower().str.strip()
    parsed = product.str.extract(r'^avo(?P<size>\d+)(?P<type>[a-z]+)(?P<grade>\d+)$')
    size = pd.to_numeric(df.get('product_category01', pd.Series(index=df.index, dtype=float)), errors='coerce')
    df['size'] = size.fillna(pd.to_numeric(parsed['size'], errors='coerce'))
    category = df.get('product_category02', pd.Series(index=df.index, dtype='string')).astype('string').str.upper().str.strip()
    category = category.replace({'NAN': pd.NA, 'NONE': pd.NA, '': pd.NA})
    df['type_code'] = category.fillna(parsed['type'].str.upper()).fillna('UNKNOWN')
    audit['non_target_or_unknown_size_rows_removed'] = int((~df['size'].isin(SIZES)).sum())
    df = df[df['size'].isin(SIZES)].copy()
    if df.empty:
        raise ValueError('No valid rows remain for sizes 32, 48, and 60.')
    df['size'] = df['size'].astype(int)
    df['week_start'] = monday_start(df['date'])
    if 'price_per_case' in df:
        df['price_per_case'] = pd.to_numeric(df['price_per_case'], errors='coerce')
    audit.update(clean_rows=len(df), date_min=str(df['date'].min().date()), date_max=str(df['date'].max().date()), companies=int(df['company'].nunique()))
    return df, audit


def weekly_series(df):
    """One chronological, gap-free series per company/size.

    Unrecorded weeks inside the observed period are assumed to have zero sales.
    They must be treated as missing observations instead if this assumption is false.
    """
    end = df['week_start'].max()
    tables = []
    for (company, size), group in df.groupby(['company', 'size'], sort=True):
        totals = group.groupby('week_start')['quantity'].sum().sort_index()
        grid = pd.date_range(totals.index.min(), end, freq='W-MON')
        table = totals.reindex(grid, fill_value=0).rename_axis('week_start').reset_index()
        table['imputed_zero_week'] = ~table['week_start'].isin(totals.index)
        table['company'], table['size'] = company, int(size)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)[['company', 'size', 'week_start', 'quantity', 'imputed_zero_week']]


def feature_row(date, history):
    if len(history) < LAGS:
        raise ValueError('At least four historical weeks are required.')
    angle = 2 * np.pi * (pd.Timestamp(date).isocalendar().week - 1) / 52.1775
    return [np.sin(angle), np.cos(angle), *[float(history[-i]) for i in range(1, 5)], float(np.mean(history[-4:]))]


def fit_model(series, trees=200):
    q = series['quantity'].to_numpy(dtype=float)
    X = [feature_row(series['week_start'].iloc[i], q[:i]) for i in range(LAGS, len(series))]
    if len(X) < 8:
        raise ValueError('Not enough historical weeks to fit a model.')
    model = RandomForestRegressor(n_estimators=trees, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(np.asarray(X), q[LAGS:])
    return model


def recursive_forecast(model, history, dates):
    """Only supplied history and previous predictions are used, never future actuals."""
    history = list(map(float, history))
    predictions = []
    for date in dates:
        prediction = max(0.0, float(model.predict(np.asarray([feature_row(date, history)]))[0]))
        predictions.append(prediction)
        history.append(prediction)
    return predictions


def backtest(weekly, folds=3, trees=200):
    rows = []
    for (company, size), series in weekly.groupby(['company', 'size'], sort=True):
        series = series.sort_values('week_start').reset_index(drop=True)
        for fold in range(folds):
            origin = len(series) - HORIZON * (folds - fold)
            if origin < 20:
                raise ValueError(f'Insufficient history for {company}/{size}: {len(series)} weeks.')
            train, test = series.iloc[:origin], series.iloc[origin:origin + HORIZON]
            model = fit_model(train, trees)
            predictions = recursive_forecast(model, train['quantity'], test['week_start'])
            for step, ((_, actual), pred) in enumerate(zip(test.iterrows(), predictions), 1):
                rows.append({'company': company, 'size': int(size), 'fold': fold + 1, 'horizon': step,
                             'train_end': train['week_start'].iloc[-1], 'week_start': actual['week_start'],
                             'actual': float(actual['quantity']), 'random_forest': pred,
                             'naive_last_week': float(train['quantity'].iloc[-1]),
                             'mean_last_4_weeks': float(train['quantity'].tail(4).mean())})
    return pd.DataFrame(rows)


def score_backtest(predictions):
    rows = []
    for (company, size), group in predictions.groupby(['company', 'size'], sort=True):
        for model in ['random_forest', 'naive_last_week', 'mean_last_4_weeks']:
            error = group['actual'] - group[model]
            rows.append({'company': company, 'size': int(size), 'model': model,
                         'mae': float(error.abs().mean()), 'rmse': float(np.sqrt((error**2).mean())),
                         'observations': len(group)})
    return pd.DataFrame(rows)


def make_forecasts(weekly, trees=200):
    rows = []
    for (company, size), series in weekly.groupby(['company', 'size'], sort=True):
        series = series.sort_values('week_start').reset_index(drop=True)
        # Refit after backtesting on ALL available history for this series.
        model = fit_model(series, trees)
        dates = pd.date_range(series['week_start'].iloc[-1] + pd.Timedelta(weeks=1), periods=HORIZON, freq='W-MON')
        values = recursive_forecast(model, series['quantity'], dates)
        for step, (date, quantity) in enumerate(zip(dates, values), 1):
            rows.append({'company': company, 'size': int(size), 'horizon': step, 'week_start': date,
                         'forecast_quantity': quantity, 'naive_last_week': float(series['quantity'].iloc[-1]),
                         'mean_last_4_weeks': float(series['quantity'].tail(4).mean())})
    return pd.DataFrame(rows)


def segment_customers(clean):
    result = clean.groupby(['company', 'customer_id'], as_index=False).agg(total_orders=('order_id', 'nunique'), total_quantity=('quantity', 'sum'))
    # Order-frequency labels are descriptive heuristics, not inferred seasonal behavior.
    result['segment'] = pd.cut(result['total_orders'], bins=[-1, 1, 2, 9, np.inf], labels=['One order', 'Two orders', 'Regular (3-9)', 'Frequent (10+)'])
    return result


def save_figures(weekly, forecasts, metrics, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for company, group in weekly.groupby('company'):
        total = group.groupby('week_start')['quantity'].sum()
        ax.plot(total.index, total.values, label=f'Company {company}', linewidth=1.4)
    ax.set(title='Weekly avocado sales by company', xlabel='Week starting Monday', ylabel='Quantity (cases)')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'weekly_sales.png', dpi=160); plt.close(fig)
    groups = list(weekly.groupby(['company', 'size'], sort=True))
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 2.7*len(groups)), squeeze=False)
    for ax, ((company, size), group) in zip(axes.flat, groups):
        fc = forecasts[(forecasts['company']==company) & (forecasts['size']==size)]
        recent = group.tail(26)
        ax.plot(recent['week_start'], recent['quantity'], label='Observed')
        ax.plot(fc['week_start'], fc['forecast_quantity'], 'o--', label='Random Forest forecast')
        ax.set(title=f'Company {company} / Size {size}', ylabel='Cases'); ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout(); fig.savefig(out/'forecasts.png', dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    means = metrics.groupby('model')['mae'].mean().sort_values()
    ax.barh(means.index, means.values, color=['#2a9d8f', '#457b9d', '#e9c46a'])
    ax.set(title='Three-fold recursive backtest', xlabel='Mean MAE across company/size series (cases)')
    fig.tight_layout(); fig.savefig(out/'backtest_comparison.png', dpi=160); plt.close(fig)


def run(input_path, output_dir, trees=200):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    clean, audit = clean_data(input_path)
    weekly = weekly_series(clean)
    predictions = backtest(weekly, trees=trees)
    metrics = score_backtest(predictions)
    forecasts = make_forecasts(weekly, trees)
    segments = segment_customers(clean)
    for name, table in [('weekly_sales', weekly), ('backtest_predictions', predictions), ('backtest_metrics', metrics), ('forecasts_4_weeks', forecasts), ('customer_segments', segments)]:
        table.to_csv(out/f'{name}.csv', index=False)
    audit.update(weekly_rows=len(weekly), zero_sales_weeks_inserted=int(weekly['imputed_zero_week'].sum()), forecast_rows=len(forecasts), backtest_rows=len(predictions), trees=trees,
                 forecast_from=str(forecasts['week_start'].min().date()), forecast_to=str(forecasts['week_start'].max().date()),
                 zero_sales_assumption='Unrecorded weeks after a series begins are treated as zero sales.',
                 partial_final_week='The final calendar week may be incomplete; no missing daily transactions can be inferred.')
    (out/'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    make_summary = {}
    for model in ['random_forest', 'naive_last_week', 'mean_last_4_weeks']:
        error = predictions['actual'] - predictions[model]
        make_summary[model] = {'pooled_mae': float(error.abs().mean()), 'pooled_rmse': float(np.sqrt((error**2).mean()))}
    (out/'evaluation_summary.json').write_text(json.dumps(make_summary, indent=2), encoding='utf-8')
    save_figures(weekly, forecasts, metrics, out)
    print(json.dumps({'audit': audit, 'backtest': make_summary}, indent=2))
    return audit, make_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data'/'demand_sales_data.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results')
    parser.add_argument('--trees', type=int, default=200)
    args = parser.parse_args()
    run(args.input, args.output_dir, args.trees)
