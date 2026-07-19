"""Prediction utilities for the Flask dashboard."""


PREDICTION_LABELS = {
    'next_day': {
        'actual': 'Actual Next Day',
        'positive': 'UP',
        'negative': 'DOWN',
    },
    'trend': {
        'actual': 'Actual 5-Day Trend',
        'positive': 'UP',
        'negative': 'DOWN',
    },
    'volatility': {
        'actual': 'Actual Volatility',
        'positive': 'HIGH',
        'negative': 'NORMAL',
    },
}

MODEL_GROUPS = {
    'ensemble': {'AdaBoost', 'Random Forest', 'Decision Tree', 'Gradient Boosting', 'Bagging'},
    'linear': {'Logistic Regression', 'SVM', 'Naive Bayes'},
}


def normalize_predictor_artifact(data):
    """Support both old and current model.pkl structures."""
    predictors = data.get('predictors')
    if predictors is not None:
        return predictors

    return {
        'next_day': {
            'name': 'Next Day Direction',
            'models': data['models'],
            'scaler': data['scaler'],
            'features': data['features'],
            'metrics': data['metrics'],
        }
    }
