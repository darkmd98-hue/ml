"""
Run this once to train all models and save them.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier,
                               GradientBoostingClassifier, BaggingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

DATASET_PATH = '../AAPL_2022_2025.csv'
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


def load_stock_csv(path):
    df = pd.read_csv(path)
    if 'Date' not in df.columns and any(str(col).endswith('Price') for col in df.columns):
        df = pd.read_csv(path, skiprows=[1, 2])
        price_col = next(col for col in df.columns if str(col).endswith('Price'))
        df.rename(columns={price_col: 'Date'}, inplace=True)
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    return df


# Load dataset
df = load_stock_csv(DATASET_PATH)

df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

df['Price_Change']   = df['Close'] - df['Open']
df['High_Low_Range'] = df['High'] - df['Low']
df['Daily_Return']   = df['Close'].pct_change()
df['MA_5']           = df['Close'].rolling(window=5).mean()
df['MA_10']          = df['Close'].rolling(window=10).mean()
df['Volatility']     = df['Close'].rolling(window=5).std()

base_df = df.drop(columns=['Date']).copy()

model_defs = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'AdaBoost':            AdaBoostClassifier(n_estimators=100, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'SVM':                 SVC(kernel='rbf', probability=True, random_state=42),
    'Naive Bayes':         GaussianNB(),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Bagging':             BaggingClassifier(n_estimators=100, random_state=42),
}

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
        m = model.__class__(**model.get_params())
        m.fit(X_train_sc, y_train_res)
        y_pred = m.predict(X_test_sc)
        model_metrics[name] = {
            'accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
            'precision': round(precision_score(y_test, y_pred, average='weighted') * 100, 2),
            'recall':    round(recall_score(y_test, y_pred, average='weighted') * 100, 2),
            'f1':        round(f1_score(y_test, y_pred, average='weighted') * 100, 2),
        }
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
