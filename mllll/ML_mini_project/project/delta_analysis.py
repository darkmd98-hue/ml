"""
Evaluate the best Next Day Direction model using a percentage delta threshold.

This script uses 2023-2025 rows as testing data, applies a +/- 2% movement
threshold, and reports the model error on meaningful UP/DOWN movements.
"""

import pickle
from pathlib import Path

import pandas as pd


DATASET_PATH = Path('../AAPL_2022_2025.csv')
MODEL_PATH = Path('model.pkl')
OUTPUT_PATH = Path('../delta_analysis_next_day.csv')
PREDICTION_TYPE = 'next_day'
DELTA_PERCENT = 2.0
TEST_START = '2023-01-01'
TEST_END = '2025-12-31'


def load_stock_csv(path):
    df = pd.read_csv(path)
    if 'Date' not in df.columns and any(str(col).endswith('Price') for col in df.columns):
        df = pd.read_csv(path, skiprows=[1, 2])
        price_col = next(col for col in df.columns if str(col).endswith('Price'))
        df.rename(columns={price_col: 'Date'}, inplace=True)
    df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
    return df


def prepare_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['Price_Change'] = df['Close'] - df['Open']
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Close'].rolling(window=5).std()
    df['Next_Close'] = df['Close'].shift(-1)
    df['Delta_Percent'] = ((df['Next_Close'] - df['Close']) / df['Close']) * 100
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


with MODEL_PATH.open('rb') as f:
    data = pickle.load(f)

predictor = data['predictors'][PREDICTION_TYPE]
metrics = predictor['metrics']
best_model_name = max(metrics, key=lambda name: metrics[name]['accuracy'])
best_model = predictor['models'][best_model_name]
scaler = predictor['scaler']
features = predictor['features']

df = prepare_features(load_stock_csv(DATASET_PATH))
test_df = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)].copy()
meaningful_df = test_df[test_df['Delta_Percent'].abs() >= DELTA_PERCENT].copy()

X = meaningful_df[features]
scaled = scaler.transform(X)
meaningful_df['Predicted_Value'] = best_model.predict(scaled)
meaningful_df['Predicted_Label'] = meaningful_df['Predicted_Value'].map({1: 'UP', 0: 'DOWN'})
meaningful_df['Actual_Value'] = (meaningful_df['Delta_Percent'] > 0).astype(int)
meaningful_df['Actual_Label'] = meaningful_df['Actual_Value'].map({1: 'UP', 0: 'DOWN'})
meaningful_df['Correct'] = meaningful_df['Predicted_Value'] == meaningful_df['Actual_Value']

output_cols = [
    'Date',
    'Close',
    'Next_Close',
    'Delta_Percent',
    'Actual_Label',
    'Predicted_Label',
    'Correct',
]
meaningful_df[output_cols].to_csv(OUTPUT_PATH, index=False)

total_rows = len(test_df)
meaningful_rows = len(meaningful_df)
correct = int(meaningful_df['Correct'].sum())
wrong = meaningful_rows - correct
accuracy = round((correct / meaningful_rows) * 100, 2) if meaningful_rows else 0
error_rate = round((wrong / meaningful_rows) * 100, 2) if meaningful_rows else 0

print('Delta Analysis: Next Day Direction')
print(f'Test period: {TEST_START} to {TEST_END}')
print(f'Delta threshold: +/- {DELTA_PERCENT}%')
print(f'Best model: {best_model_name}')
print(f'Original test rows: {total_rows}')
print(f'Meaningful movement rows: {meaningful_rows}')
print(f'Correct predictions: {correct}')
print(f'Wrong predictions: {wrong}')
print(f'Accuracy after delta: {accuracy}%')
print(f'Error rate after delta: {error_rate}%')
print(f'Results saved to: {OUTPUT_PATH}')
