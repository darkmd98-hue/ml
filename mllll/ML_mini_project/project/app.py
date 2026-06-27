from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from scipy import stats

app = Flask(__name__)

DATASET_PATH = '../AAPL_2022_2025.csv'


def load_stock_csv(path):
    df = pd.read_csv(path)
    if 'Date' not in df.columns and 'Price' in df.columns:
        df = pd.read_csv(path, skiprows=[1, 2])
        df.rename(columns={'Price': 'Date'}, inplace=True)
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    return df

# Load trained models
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

models   = data['models']
scaler   = data['scaler']
features = data['features']
metrics  = data['metrics']

# Load and prepare the dataset for date lookup
df_raw = load_stock_csv(DATASET_PATH)
df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df_raw.sort_values('Date', inplace=True)
df_raw.reset_index(drop=True, inplace=True)

df_raw['Price_Change']   = df_raw['Close'] - df_raw['Open']
df_raw['High_Low_Range'] = df_raw['High'] - df_raw['Low']
df_raw['Daily_Return']   = df_raw['Close'].pct_change()
df_raw['MA_5']           = df_raw['Close'].rolling(window=5).mean()
df_raw['MA_10']          = df_raw['Close'].rolling(window=10).mean()
df_raw['Volatility']     = df_raw['Close'].rolling(window=5).std()
df_raw['Target']         = (df_raw['Close'].shift(-1) > df_raw['Close']).astype(int)
df_raw.dropna(inplace=True)
df_raw.reset_index(drop=True, inplace=True)

# Available dates for the dropdown
available_dates = df_raw['Date'].dt.strftime('%Y-%m-%d').tolist()


@app.route('/')
def index():
    latest = df_raw['Date'].dt.strftime('%Y-%m-%d').iloc[-1]
    return render_template('index.html', latest_date=latest, metrics=metrics)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload     = request.get_json()
        chosen_date = pd.to_datetime(payload.get('date'))

        # normalize to date only (strip time component)
        chosen_date = chosen_date.normalize()
        row = df_raw[df_raw['Date'].dt.normalize() == chosen_date]
        if row.empty:
            return jsonify({'error': f'Date {chosen_date.date()} not found in dataset'}), 400

        row = row.iloc[0]
        values = [row[f] for f in features]
        scaled = scaler.transform([values])

        actual = int(row['Target'])
        results = []
        for name, model in models.items():
            pred  = model.predict(scaled)[0]
            proba = model.predict_proba(scaled)[0] if hasattr(model, 'predict_proba') else None
            conf  = round(float(max(proba)) * 100, 2) if proba is not None else None
            results.append({
                'name':       name,
                'prediction': int(pred),
                'label':      'UP' if pred == 1 else 'DOWN',
                'correct':    int(pred) == actual,
                'confidence': conf,
                'accuracy':   metrics[name]['accuracy'],
                'precision':  metrics[name]['precision'],
                'recall':     metrics[name]['recall'],
                'f1':         metrics[name]['f1'],
            })

        results.sort(key=lambda x: x['accuracy'], reverse=True)

        stock_info = {
            'date':   row['Date'].strftime('%b %d, %Y'),
            'open':   round(float(row['Open']), 2),
            'high':   round(float(row['High']), 2),
            'low':    round(float(row['Low']), 2),
            'close':  round(float(row['Close']), 2),
            'volume': int(row['Volume']),
            'actual': 'UP' if actual == 1 else 'DOWN',
        }

        return jsonify({'results': results, 'stock': stock_info})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
