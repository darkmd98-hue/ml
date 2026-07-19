"""Reusable preprocessing for the Apple stock dataset.

The project receives a CSV exported with extra metadata rows. This module keeps
the cleanup logic in one place so the Flask app, training script, and analysis
reports all read the same normalized data.
"""

from pathlib import Path

import pandas as pd


DATASET_PATH = Path('../AAPL_2022_2025.csv')
EXPECTED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']


def _standardize_column_name(column):
    name = str(column).strip()
    if name.endswith('Price'):
        return 'Date'
    return name


def load_stock_csv(path=DATASET_PATH):
    """Load the Apple CSV and remove unnecessary header/metadata rows."""
    df = pd.read_csv(path)
    df.columns = [_standardize_column_name(col) for col in df.columns]

    has_metadata_rows = 'Date' in df.columns and str(df['Date'].iloc[0]).strip() in {'Ticker', 'Date'}
    if 'Date' not in df.columns or has_metadata_rows:
        df = pd.read_csv(path, skiprows=[1, 2])
        df.columns = [_standardize_column_name(col) for col in df.columns]

    df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
    return df


def preprocess_stock_data(path=DATASET_PATH, drop_missing=True):
    """Return a chronologically sorted stock DataFrame with clean columns.

    Steps:
    - loads the Apple dataset
    - removes the extra yfinance-style header rows
    - standardizes column names
    - converts Date to datetime
    - sorts records by Date
    - removes duplicate dates
    - handles missing values when requested
    """
    df = load_stock_csv(path).copy()
    df.columns = [_standardize_column_name(col) for col in df.columns]

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f'Missing required columns: {", ".join(missing_columns)}')

    df = df[[col for col in df.columns if str(col).strip()]]
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
    df.reset_index(drop=True, inplace=True)

    numeric_columns = [col for col in EXPECTED_COLUMNS if col != 'Date']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if drop_missing:
        df.dropna(subset=EXPECTED_COLUMNS, inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def get_preprocessing_report(df):
    """Return simple data quality information for documentation/debugging."""
    return {
        'rows': len(df),
        'duplicate_dates': int(df.duplicated(subset=['Date']).sum()) if 'Date' in df else None,
        'missing_values': df.isna().sum().to_dict(),
        'columns': list(df.columns),
    }
