from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

from feature_engineering import add_classification_targets, add_market_features
from prediction import MODEL_GROUPS, PREDICTION_LABELS, normalize_predictor_artifact
from preprocessing import DATASET_PATH, preprocess_stock_data

app = Flask(__name__)

# Load trained models
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

predictors = normalize_predictor_artifact(data)

# Load and prepare the dataset for date lookup
df_raw = preprocess_stock_data(DATASET_PATH)
df_raw = add_market_features(df_raw)
df_raw = add_classification_targets(df_raw)
df_raw.dropna(inplace=True)
df_raw.reset_index(drop=True, inplace=True)

# Available dates for the dropdown
available_dates = df_raw['Date'].dt.strftime('%Y-%m-%d').tolist()


@app.route('/')
def index():
    latest = df_raw['Date'].dt.strftime('%Y-%m-%d').iloc[-1]
    return render_template('index.html', latest_date=latest)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload     = request.get_json()
        chosen_date = pd.to_datetime(payload.get('date'))
        prediction_type = payload.get('type', 'next_day')
        focus = payload.get('focus', 'all')
        if prediction_type not in predictors:
            return jsonify({'error': f'Prediction type {prediction_type} is not available. Retrain the model first.'}), 400

        # normalize to date only (strip time component)
        chosen_date = chosen_date.normalize()
        row = df_raw[df_raw['Date'].dt.normalize() == chosen_date]
        if row.empty:
            return jsonify({'error': f'Date {chosen_date.date()} not found in dataset'}), 400

        row = row.iloc[0]
        predictor = predictors[prediction_type]
        models = predictor['models']
        scaler = predictor['scaler']
        features = predictor['features']
        metrics = predictor['metrics']
        labels = PREDICTION_LABELS[prediction_type]
        target_col = f'Target_{prediction_type}'
        values = pd.DataFrame([[row[f] for f in features]], columns=features)
        scaled = scaler.transform(values)

        actual = int(row[target_col])
        results = []
        for name, model in models.items():
            if focus != 'all' and name not in MODEL_GROUPS.get(focus, set()):
                continue
            pred  = model.predict(scaled)[0]
            proba = model.predict_proba(scaled)[0] if hasattr(model, 'predict_proba') else None
            conf  = round(float(max(proba)) * 100, 2) if proba is not None else None
            results.append({
                'name':       name,
                'prediction': int(pred),
                'label':      labels['positive'] if pred == 1 else labels['negative'],
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
            'actual': labels['positive'] if actual == 1 else labels['negative'],
            'actual_label': labels['actual'],
            'positive_label': labels['positive'],
            'negative_label': labels['negative'],
            'prediction_type': predictor['name'],
        }

        return jsonify({'results': results, 'stock': stock_info})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
