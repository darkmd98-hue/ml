"""
Run this once to train all models and save them.
Usage: python train_model.py
"""

import numpy as np
import pickle
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from evaluation import classification_metrics
from feature_engineering import add_market_features
from model_training import clone_model, get_classification_models
from preprocessing import DATASET_PATH, preprocess_stock_data

SPLIT_OUTPUT_DIR = Path('../train_test_data')
PREDICTION_TYPES = {
    'next_day': {
        'name': 'Next Day Direction',
        'target': lambda df: (df['Close'].shift(-1) > df['Close']).astype(int),
    },
    'trend': {
        'name': 'Short-term Trend',
        'target': lambda df: (df['Close'].shift(-5) > df['Close']).astype(int),
    },
    'volatility': {
        'name': 'High Volatility Day',
        'target': lambda df: (
            (df['High'].shift(-1) - df['Low'].shift(-1)) >
            df['High_Low_Range'].rolling(window=20, min_periods=5).median()
        ).astype(int),
    },
}


# Load dataset
df = preprocess_stock_data(DATASET_PATH)
df = add_market_features(df)

base_df = df.drop(columns=['Date']).copy()

model_defs = get_classification_models()

predictors = {}
SPLIT_OUTPUT_DIR.mkdir(exist_ok=True)

for type_key, config in PREDICTION_TYPES.items():
    task_df = base_df.copy()
    task_df['Target'] = config['target'](df)
    task_df.dropna(inplace=True)
    task_df.reset_index(drop=True, inplace=True)

    z_scores = np.abs(stats.zscore(task_df.select_dtypes(include='number')))
    mask = (z_scores < 3).all(axis=1)
    df_clean = task_df[mask].reset_index(drop=True)

    X_final = df_clean.drop('Target', axis=1)
    y_final = df_clean['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
    )

    train_df = X_train.copy()
    train_df['Target'] = y_train
    test_df = X_test.copy()
    test_df['Target'] = y_test
    train_df.to_csv(SPLIT_OUTPUT_DIR / f'{type_key}_train.csv', index=False)
    test_df.to_csv(SPLIT_OUTPUT_DIR / f'{type_key}_test.csv', index=False)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc  = scaler.transform(X_test)

    trained_models = {}
    model_metrics  = {}

    print(f'\n{config["name"]}')
    for name, model in model_defs.items():
        m = clone_model(model)
        m.fit(X_train_sc, y_train_res)
        y_pred = m.predict(X_test_sc)
        model_metrics[name] = classification_metrics(y_test, y_pred)
        trained_models[name] = m
        print(f'{name}: accuracy={model_metrics[name]["accuracy"]}%')

    predictors[type_key] = {
        'name': config['name'],
        'models': trained_models,
        'scaler': scaler,
        'features': list(X_final.columns),
        'metrics': model_metrics,
    }

with open('model.pkl', 'wb') as f:
    pickle.dump({'predictors': predictors}, f)

print('\nAll models saved to model.pkl')
