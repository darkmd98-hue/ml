"""
Create one monthly actual-vs-predicted graph for each next-day price model.

The report uses the latest complete month available in the dataset, trains on
data before 2023, predicts next-day close prices, and marks +/- 2% error.
"""

from html import escape
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from feature_engineering import PRICE_FEATURES, prepare_price_prediction_data
from model_training import get_price_regression_models
from preprocessing import DATASET_PATH, preprocess_stock_data
from visualization import polyline, scale_points

REPORT_PATH = Path('../monthly_model_graphs.html')
DETAILS_PATH = Path('../monthly_model_graphs.csv')
TEST_START = '2023-01-01'
TOLERANCE_PERCENT = 2.0


def make_chart(model_name, month_df):
    width, height, padding = 760, 300, 44
    actual = month_df['Next_Close'].tolist()
    predicted = month_df['Predicted_Next_Close'].tolist()
    min_value = min(actual + predicted)
    max_value = max(actual + predicted)
    margin = (max_value - min_value) * 0.08 or 1
    min_value -= margin
    max_value += margin

    actual_points = scale_points(actual, width, height, padding, min_value, max_value)
    predicted_points = scale_points(predicted, width, height, padding, min_value, max_value)
    error_bars = []
    max_error = max(month_df['Error_Percent'].max(), TOLERANCE_PERCENT, 1)
    bar_base = height - 24
    bar_width = max((width - 2 * padding) / max(len(month_df), 1) - 4, 5)

    for idx, row in month_df.reset_index(drop=True).iterrows():
        x = padding + (idx / max(len(month_df) - 1, 1)) * (width - 2 * padding)
        bar_height = (row['Error_Percent'] / max_error) * 54
        color = '#1f883d' if row['Within_2_Percent'] else '#cf222e'
        error_bars.append(
            f'<rect x="{x - bar_width / 2:.2f}" y="{bar_base - bar_height:.2f}" '
            f'width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}" opacity=".78"/>'
        )

    dates = [d.strftime('%d') for d in month_df['Date']]
    tick_labels = []
    for idx, day in enumerate(dates):
        if idx % 3 == 0 or idx == len(dates) - 1:
            x = padding + (idx / max(len(dates) - 1, 1)) * (width - 2 * padding)
            tick_labels.append(f'<text x="{x:.2f}" y="292" text-anchor="middle">{day}</text>')

    return f"""
      <section class="model-card">
        <h2>{escape(model_name)}</h2>
        <div class="meta">
          <span>MAPE: {month_df['Model_MAPE'].iloc[0]:.2f}%</span>
          <span>Within +/-2%: {month_df['Within_2_Percent'].mean() * 100:.2f}%</span>
          <span>Rows: {len(month_df)}</span>
        </div>
        <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(model_name)} actual versus predicted chart">
          <rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="#ffffff"/>
          <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#d8dee8"/>
          <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#d8dee8"/>
          <text x="12" y="28">${max_value:.2f}</text>
          <text x="12" y="{height - padding + 4}">${min_value:.2f}</text>
          {''.join(error_bars)}
          <polyline points="{polyline(actual_points)}" fill="none" stroke="#0969da" stroke-width="3"/>
          <polyline points="{polyline(predicted_points)}" fill="none" stroke="#cf222e" stroke-width="3" stroke-dasharray="7 5"/>
          {''.join(tick_labels)}
        </svg>
      </section>
    """


df = prepare_price_prediction_data(preprocess_stock_data(DATASET_PATH))
features = PRICE_FEATURES

train_df = df[df['Date'] < TEST_START].copy()
test_df = df[df['Date'] >= TEST_START].copy()
latest_month = test_df['Date'].dt.to_period('M').max()
month_df = test_df[test_df['Date'].dt.to_period('M') == latest_month].copy()

X_train = train_df[features]
y_train = train_df['Next_Close']
X_month = month_df[features]
y_month = month_df['Next_Close']

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_month_sc = scaler.transform(X_month)

models = get_price_regression_models()

all_results = []
cards = []

for model_name, model in models.items():
    model.fit(X_train_sc, y_train)
    predictions = model.predict(X_month_sc)
    model_df = month_df[['Date', 'Close', 'Next_Close']].copy()
    model_df['Model'] = model_name
    model_df['Predicted_Next_Close'] = predictions
    model_df['Error_Percent'] = (
        (model_df['Predicted_Next_Close'] - model_df['Next_Close']).abs() /
        model_df['Next_Close']
    ) * 100
    model_df['Within_2_Percent'] = model_df['Error_Percent'] <= TOLERANCE_PERCENT
    model_df['Model_MAPE'] = mean_absolute_percentage_error(y_month, predictions) * 100
    all_results.append(model_df)
    cards.append(make_chart(model_name, model_df))

result_df = pd.concat(all_results, ignore_index=True)
result_df.to_csv(DETAILS_PATH, index=False)

report = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Monthly Model Graphs</title>
  <style>
    body{{font-family:Segoe UI,Arial,sans-serif;background:#f5f7fb;color:#111827;margin:0;padding:28px}}
    main{{max-width:980px;margin:0 auto}}
    h1{{margin:0 0 8px;font-size:1.8rem}}
    p{{margin:0 0 20px;color:#5f6b7a}}
    .legend{{display:flex;gap:16px;margin:12px 0 24px;flex-wrap:wrap;color:#374151;font-weight:600}}
    .legend span{{display:inline-flex;align-items:center;gap:6px}}
    .line{{width:30px;height:3px;display:inline-block;background:#0969da}}
    .dash{{width:30px;height:0;border-top:3px dashed #cf222e;display:inline-block}}
    .ok{{width:14px;height:14px;background:#1f883d;display:inline-block}}
    .bad{{width:14px;height:14px;background:#cf222e;display:inline-block}}
    .model-card{{background:#fff;border:1px solid #d8dee8;border-radius:8px;padding:20px;margin-bottom:18px}}
    .model-card h2{{font-size:1.1rem;margin:0 0 8px}}
    .meta{{display:flex;gap:12px;flex-wrap:wrap;color:#5f6b7a;font-size:.9rem;margin-bottom:10px}}
    svg{{width:100%;height:auto;display:block}}
    svg text{{font-size:12px;fill:#5f6b7a}}
  </style>
</head>
<body>
<main>
  <h1>Actual vs Predicted Next-Day Close</h1>
  <p>Month: {latest_month.strftime('%B %Y')} | Tolerance: +/- {TOLERANCE_PERCENT}% | Training data: before {TEST_START}</p>
  <div class="legend">
    <span><i class="line"></i> Actual next close</span>
    <span><i class="dash"></i> Predicted next close</span>
    <span><i class="ok"></i> Error within +/-2%</span>
    <span><i class="bad"></i> Error above +/-2%</span>
  </div>
  {''.join(cards)}
</main>
</body>
</html>
"""

REPORT_PATH.write_text(report, encoding='utf-8')

print('Monthly model graphs generated')
print(f'Month: {latest_month.strftime("%B %Y")}')
print(f'Models graphed: {len(models)}')
print(f'HTML report: {REPORT_PATH}')
print(f'Detail CSV: {DETAILS_PATH}')
