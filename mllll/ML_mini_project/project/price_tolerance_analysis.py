"""
Predict next-day close price and evaluate predictions with a +/- 2% tolerance.

This matches a regression-style guide requirement:
if the predicted next close is within +/- 2% of the actual next close, the
prediction is treated as acceptable; otherwise it is counted as an error.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


DATASET_PATH = Path('../AAPL_2022_2025.csv')
OUTPUT_PATH = Path('../price_tolerance_analysis.csv')
TOLERANCE_PERCENT = 2.0
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


def prepare_dataset(df):
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
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


df = prepare_dataset(load_stock_csv(DATASET_PATH))

features = [
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

train_df = df[df['Date'] < TEST_START].copy()
test_df = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)].copy()

X_train = train_df[features]
y_train = train_df['Next_Close']
X_test = test_df[features]
y_test = test_df['Next_Close']

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=200, random_state=42),
    'SVR': SVR(kernel='rbf'),
}

model_scores = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    predictions[name] = y_pred
    model_scores[name] = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mape': mean_absolute_percentage_error(y_test, y_pred) * 100,
        'r2': r2_score(y_test, y_pred),
    }

best_model_name = min(model_scores, key=lambda name: model_scores[name]['mape'])
best_pred = predictions[best_model_name]

result_df = test_df[['Date', 'Close', 'Next_Close']].copy()
result_df['Predicted_Next_Close'] = best_pred
result_df['Error_Percent'] = (
    (result_df['Predicted_Next_Close'] - result_df['Next_Close']).abs() /
    result_df['Next_Close']
) * 100
result_df['Within_2_Percent'] = result_df['Error_Percent'] <= TOLERANCE_PERCENT
result_df.to_csv(OUTPUT_PATH, index=False)

total = len(result_df)
within = int(result_df['Within_2_Percent'].sum())
outside = total - within
within_rate = round((within / total) * 100, 2) if total else 0
error_rate = round((outside / total) * 100, 2) if total else 0

print('Next Day Price Tolerance Analysis')
print(f'Training period: before {TEST_START}')
print(f'Testing period: {TEST_START} to {TEST_END}')
print(f'Tolerance: +/- {TOLERANCE_PERCENT}%')
print(f'Best model: {best_model_name}')
print(f'MAE: {model_scores[best_model_name]["mae"]:.4f}')
print(f'MAPE: {model_scores[best_model_name]["mape"]:.2f}%')
print(f'R2 score: {model_scores[best_model_name]["r2"]:.4f}')
print(f'Total test rows: {total}')
print(f'Within tolerance: {within}')
print(f'Outside tolerance: {outside}')
print(f'Acceptable prediction rate: {within_rate}%')
print(f'Error rate: {error_rate}%')
print(f'Results saved to: {OUTPUT_PATH}')
