"""Feature engineering used across StockSense AI.

The formulas intentionally match the existing project logic.
"""

CLASSIFICATION_FEATURES = [
    'Open',
    'High',
    'Low',
    'Close',
    'Volume',
    'Price_Change',
    'High_Low_Range',
    'Daily_Return',
    'MA_5',
    'MA_10',
    'Volatility',
]

PRICE_FEATURES = CLASSIFICATION_FEATURES.copy()


def add_market_features(df, include_previous_close=False, include_next_close=False, drop_missing=False):
    """Add engineered market features without changing existing formulas."""
    df = df.copy()
    df['Price_Change'] = df['Close'] - df['Open']
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Close'].rolling(window=5).std()

    if include_previous_close:
        df['Previous_Close'] = df['Close'].shift(1)
    if include_next_close:
        df['Next_Close'] = df['Close'].shift(-1)
    if drop_missing:
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def add_classification_targets(df):
    """Add all classification targets used by the Flask prediction dashboard."""
    df = df.copy()
    df['Target_next_day'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['Target_trend'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    df['Target_volatility'] = (
        (df['High'].shift(-1) - df['Low'].shift(-1)) >
        df['High_Low_Range'].rolling(window=20, min_periods=5).median()
    ).astype(int)
    return df


def add_delta_features(df):
    """Add next-close and percentage movement for delta analysis."""
    df = add_market_features(df, include_next_close=True)
    df['Delta_Percent'] = ((df['Next_Close'] - df['Close']) / df['Close']) * 100
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def prepare_price_prediction_data(df):
    """Add features and Next_Close target for regression scripts."""
    return add_market_features(df, include_next_close=True, drop_missing=True)
