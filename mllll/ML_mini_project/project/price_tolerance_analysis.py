"""
Predict next-day close price and evaluate predictions with a +/- 2% tolerance.

This matches a regression-style guide requirement:
if the predicted next close is within +/- 2% of the actual next close, the
prediction is treated as acceptable; otherwise it is counted as an error.
"""

from pathlib import Path

from sklearn.preprocessing import StandardScaler

from evaluation import regression_metrics, within_tolerance_rate
from feature_engineering import PRICE_FEATURES, prepare_price_prediction_data
from model_training import get_price_regression_models
from preprocessing import DATASET_PATH, preprocess_stock_data

OUTPUT_PATH = Path('../price_tolerance_analysis.csv')
TOLERANCE_PERCENT = 2.0
TEST_START = '2023-01-01'
TEST_END = '2025-12-31'


df = prepare_price_prediction_data(preprocess_stock_data(DATASET_PATH))
features = PRICE_FEATURES

train_df = df[df['Date'] < TEST_START].copy()
test_df = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)].copy()

X_train = train_df[features]
y_train = train_df['Next_Close']
X_test = test_df[features]
y_test = test_df['Next_Close']

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

models = get_price_regression_models()

model_scores = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    predictions[name] = y_pred
    model_scores[name] = regression_metrics(y_test, y_pred)

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

rates = within_tolerance_rate(result_df['Error_Percent'], TOLERANCE_PERCENT)

print('Next Day Price Tolerance Analysis')
print(f'Training period: before {TEST_START}')
print(f'Testing period: {TEST_START} to {TEST_END}')
print(f'Tolerance: +/- {TOLERANCE_PERCENT}%')
print(f'Best model: {best_model_name}')
print(f'MAE: {model_scores[best_model_name]["mae"]:.4f}')
print(f'MAPE: {model_scores[best_model_name]["mape"]:.2f}%')
print(f'R2 score: {model_scores[best_model_name]["r2"]:.4f}')
print(f'Total test rows: {rates["total"]}')
print(f'Within tolerance: {rates["within"]}')
print(f'Outside tolerance: {rates["outside"]}')
print(f'Acceptable prediction rate: {rates["within_rate"]}%')
print(f'Error rate: {rates["error_rate"]}%')
print(f'Results saved to: {OUTPUT_PATH}')
