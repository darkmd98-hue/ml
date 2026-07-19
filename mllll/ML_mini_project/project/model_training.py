"""Model definitions and training helpers for StockSense AI."""

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def get_classification_models():
    """Return the classification algorithms used by the dashboard."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Bagging': BaggingClassifier(n_estimators=100, random_state=42),
    }


def get_price_regression_models():
    """Return regression models for next-day price prediction analysis."""
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=200, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'SVR': SVR(kernel='rbf'),
    }


def get_three_month_graph_models():
    """Return the four models requested for the multi-model monthly graph."""
    return {
        'RF': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVM': SVR(kernel='rbf'),
        'DT': DecisionTreeRegressor(random_state=42),
        'LR': LinearRegression(),
    }


def clone_model(model):
    """Clone a scikit-learn model without changing its configured parameters."""
    return model.__class__(**model.get_params())
