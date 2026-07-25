"""
Create 3 monthly graphs with multiple model predictions in each graph.

Each graph contains:
- Actual next-day closing price from the dataset
- Random Forest prediction
- SVM/SVR prediction
- Decision Tree prediction
- LR prediction, using Linear Regression for numeric price prediction
"""

from html import escape
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from feature_engineering import PRICE_FEATURES, prepare_price_prediction_data
from model_training import get_three_month_graph_models
from preprocessing import DATASET_PATH, preprocess_stock_data
from visualization import make_svg_line

REPORT_PATH = Path('../three_month_model_graphs.html')
DETAILS_PATH = Path('../three_month_model_graphs.csv')
TEST_START = '2023-01-01'


def make_month_chart(month_period, month_df):
    width, height, padding = 900, 380, 58
    series_columns = ['Actual', 'RF', 'SVM', 'DT', 'LR']
    all_values = []
    for col in series_columns:
        all_values.extend(month_df[col].tolist())

    min_value = min(all_values)
    max_value = max(all_values)
    margin = (max_value - min_value) * 0.08 or 1
    min_value -= margin
    max_value += margin

    colors = {
        'Actual': '#111827',
        'RF': '#f97316',
        'SVM': '#22c55e',
        'DT': '#38bdf8',
        'LR': '#a855f7',
    }
    lines = []
    label_positions = []
    span = max_value - min_value or 1
    x_end = width - padding + 5
    for col in series_columns:
        lines.append(make_svg_line(month_df[col].tolist(), colors[col], width, height, padding, min_value, max_value))
        last_val = month_df[col].iloc[-1]
        y_end = padding + (max_value - last_val) / span * (height - 2 * padding)
        label_positions.append({'col': col, 'y': y_end})
    
    label_positions.sort(key=lambda item: item['y'])
    
    min_dist = 14
    for i in range(1, len(label_positions)):
        if label_positions[i]['y'] - label_positions[i-1]['y'] < min_dist:
            label_positions[i]['y'] = label_positions[i-1]['y'] + min_dist
            
    line_labels = []
    for pos in label_positions:
        col = pos['col']
        y_adj = pos['y']
        line_labels.append(f'<text x="{x_end}" y="{y_adj + 4}" fill="{colors[col]}" font-weight="bold">{col}</text>')

    tick_labels = []
    days = [d.strftime('%d') for d in month_df['Date']]
    for idx, day in enumerate(days):
        if idx % 3 == 0 or idx == len(days) - 1:
            x = padding + (idx / max(len(days) - 1, 1)) * (width - 2 * padding)
            tick_labels.append(f'<text x="{x:.2f}" y="{height - 16}" text-anchor="middle">{day}</text>')

    model_metrics = []
    for col in ['RF', 'SVM', 'DT', 'LR']:
        mape = mean_absolute_percentage_error(month_df['Actual'], month_df[col]) * 100
        model_metrics.append(f'<span>{col} MAPE: {mape:.2f}%</span>')

    legend_items = [
        ('Actual', colors['Actual']),
        ('RF - Random Forest', colors['RF']),
        ('SVM - Support Vector Machine', colors['SVM']),
        ('DT - Decision Tree', colors['DT']),
        ('LR - Linear Regression', colors['LR']),
    ]
    legend_html = ''.join(
        f'<span><i style="background:{color}"></i>{escape(label)}</span>'
        for label, color in legend_items
    )

    return f"""
      <section class="chart-card">
        <h2>{escape(month_period.strftime('%B %Y'))}</h2>
        <div class="meta">{''.join(model_metrics)}</div>
        <div class="legend" style="margin-top: 10px; margin-bottom: 20px;">{legend_html}</div>
        <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(month_period.strftime('%B %Y'))} multi-model prediction graph">
          <rect x="0" y="0" width="{width}" height="{height}" rx="12" fill="#ffffff"/>
          <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#d8dee8"/>
          <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#d8dee8"/>
          <text x="12" y="32">${max_value:.2f}</text>
          <text x="12" y="{height - padding + 4}">${min_value:.2f}</text>
          {''.join(lines)}
          {''.join(line_labels)}
          {''.join(tick_labels)}
        </svg>
      </section>
    """


df = prepare_price_prediction_data(preprocess_stock_data(DATASET_PATH))
features = PRICE_FEATURES

train_df = df[df['Date'] < TEST_START].copy()
test_df = df[df['Date'] >= TEST_START].copy()
latest_three_months = sorted(test_df['Date'].dt.to_period('M').drop_duplicates().tail(3).tolist())

X_train = train_df[features]
y_train = train_df['Next_Close']

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

models = get_three_month_graph_models()

trained_models = {}
for label, model in models.items():
    model.fit(X_train_sc, y_train)
    trained_models[label] = model

all_rows = []
cards = []

for month_period in latest_three_months:
    month_df = test_df[test_df['Date'].dt.to_period('M') == month_period].copy()
    X_month_sc = scaler.transform(month_df[features])

    graph_df = month_df[['Date', 'Close', 'Next_Close']].copy()
    graph_df.rename(columns={'Next_Close': 'Actual'}, inplace=True)
    for label, model in trained_models.items():
        graph_df[label] = model.predict(X_month_sc)
    graph_df['Month'] = month_period.strftime('%Y-%m')
    all_rows.append(graph_df[['Month', 'Date', 'Close', 'Actual', 'RF', 'SVM', 'DT', 'LR']])
    cards.append(make_month_chart(month_period, graph_df))

result_df = pd.concat(all_rows, ignore_index=True)
result_df.to_csv(DETAILS_PATH, index=False)

report = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Three Month Multi-Model Graphs</title>
  <style>
    body{{font-family:Segoe UI,Arial,sans-serif;background:#f5f7fb;color:#111827;margin:0;padding:28px}}
    main{{max-width:1080px;margin:0 auto}}
    h1{{font-size:1.9rem;margin:0 0 8px}}
    p{{color:#5f6b7a;margin:0 0 20px}}
    .legend{{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0 24px;font-weight:600;color:#374151}}
    .legend span{{display:inline-flex;gap:7px;align-items:center}}
    .legend i{{width:28px;height:4px;border-radius:999px;display:inline-block}}
    .chart-card{{background:#fff;border:1px solid #d8dee8;border-radius:8px;padding:20px;margin-bottom:20px}}
    .chart-card h2{{font-size:1.15rem;margin:0 0 8px}}
    .meta{{display:flex;gap:12px;flex-wrap:wrap;color:#5f6b7a;font-size:.9rem;margin-bottom:10px}}
    svg{{display:block;width:100%;height:auto}}
    svg text{{font-size:12px;fill:#5f6b7a}}
  </style>
</head>
<body>
<main>
  <h1>Three-Month Actual vs Model Predictions</h1>
  <p>Each graph shows one month. Actual next-day close is plotted with four model predictions in the same graph.</p>
  {''.join(cards)}
</main>
</body>
</html>
"""

REPORT_PATH.write_text(report, encoding='utf-8')

print('Three-month multi-model graphs generated')
print('Months:', ', '.join(month.strftime('%B %Y') for month in latest_three_months))
print(f'HTML report: {REPORT_PATH}')
print(f'Detail CSV: {DETAILS_PATH}')
