import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

sys.path.append(os.path.abspath('project'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier,
                               GradientBoostingClassifier, BaggingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

from preprocessing import preprocess_stock_data
from feature_engineering import add_classification_targets, CLASSIFICATION_FEATURES, add_market_features

import warnings
warnings.filterwarnings('ignore')

# 1. Load Apple dataset via preprocessing.py
df_raw = preprocess_stock_data('AAPL_2022_2025.csv')
df_market = add_market_features(df_raw)
df = add_classification_targets(df_market)

# 2. Print row counts for all three classification targets (Target_next_day, Target_trend, Target_volatility)
print("=== Apple Dataset (AAPL_2022_2025.csv) - Dates: 2022-01-03 to 2025-12-30 ===")
print("Data source is likely yfinance, as evidenced by the header structure in the CSV and the explicit reference in preprocessing.py ('removes the extra yfinance-style header rows').\n")

print("=== Train/Test Split Row Counts (80:20 Split) ===")
targets = ['Target_next_day', 'Target_trend', 'Target_volatility']

# We need to process each target through the outlier removal to get the final row counts exactly like the pipeline
# The pipeline drops missing target values. The feature engineering shifts values, which creates missing values at the end.
for target in targets:
    df_t = df.copy()
    
    # Need all features for outlier detection
    # First, calculate technical features (since CLASSIFICATION_FEATURES uses them)
    
    df_t.dropna(subset=CLASSIFICATION_FEATURES + [target], inplace=True)
    
    X_t = df_t[CLASSIFICATION_FEATURES]
    y_t = df_t[target]
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_t, y_t)
    
    df_combined = pd.DataFrame(X_res, columns=X_t.columns)
    df_combined['Target'] = y_res.values
    
    z_scores = np.abs(stats.zscore(df_combined.select_dtypes(include='number')))
    mask = (z_scores < 3).all(axis=1)
    df_clean = df_combined[mask].reset_index(drop=True)
    
    X_final_t = df_clean.drop('Target', axis=1)
    y_final_t = df_clean['Target']
    
    X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(
        X_final_t, y_final_t, test_size=0.2, random_state=42, stratify=y_final_t
    )
    
    print(f"Target '{target}': Train rows = {len(X_tr_t)}, Test rows = {len(X_te_t)}")

print("\n")

# 3. Proceed with the main pipeline for Target_next_day to regenerate graphs
df_main = df.copy()

df_main.dropna(subset=CLASSIFICATION_FEATURES + ['Target_next_day'], inplace=True)
X = df_main[CLASSIFICATION_FEATURES]
y = df_main['Target_next_day']

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

df_combined = pd.DataFrame(X_res, columns=X.columns)
df_combined['Target'] = y_res.values
z_scores = np.abs(stats.zscore(df_combined.select_dtypes(include='number')))
mask = (z_scores < 3).all(axis=1)
df_clean = df_combined[mask].reset_index(drop=True)

X_final = df_clean.drop('Target', axis=1)
y_final = df_clean['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'AdaBoost':            AdaBoostClassifier(n_estimators=100, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'SVM':                 SVC(kernel='rbf', random_state=42),
    'Naive Bayes':         GaussianNB(),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Bagging':             BaggingClassifier(n_estimators=100, random_state=42)
}

print("=== Classification Model Performance (Apple Dataset) ===")
metrics_list = []
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    cm = confusion_matrix(y_test, y_pred)
    pred_down = cm[0][0] + cm[1][0]
    pred_up = cm[0][1] + cm[1][1]
    bias = "Up" if pred_up > pred_down else "Down" if pred_down > pred_up else "Balanced"
    
    metrics_list.append({
        'Algorithm': name,
        'Accuracy':  round(acc * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall':    round(rec * 100, 2),
        'F1-Score':  round(f1 * 100, 2)
    })
    
    print(f"Model: {name}")
    print(f"  Accuracy: {acc*100:.2f}%, Precision: {prec*100:.2f}%, Recall: {rec*100:.2f}%, F1: {f1*100:.2f}%")
    print(f"  Predicted Down: {pred_down}, Predicted Up: {pred_up} => Biased toward: {bias}")
    
    # Save confusion matrix image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {name} (Apple)')
    plt.tight_layout()
    plt.savefig(f'{name.replace(" ", "_").lower()}_cm.png', dpi=150)
    plt.close()

metrics_df = pd.DataFrame(metrics_list).sort_values('Accuracy', ascending=False).reset_index(drop=True)

# Generate accuracy bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(metrics_df['Algorithm'], metrics_df['Accuracy'],
               color=plt.cm.tab10.colors[:len(metrics_df)], edgecolor='black')
plt.ylim(0, 115)
plt.ylabel('Accuracy (%)')
plt.title('Apple Stock - Algorithm Accuracy Comparison')
plt.xticks(rotation=30, ha='right')
for bar, val in zip(bars, metrics_df['Accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('apple_accuracy_bar_chart.png', dpi=150, bbox_inches='tight')
plt.close()

# Generate performance comparison table
fig, ax = plt.subplots(figsize=(13, 4))
ax.axis('off')

col_labels = ['Algorithm', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']
table_data = metrics_df.values.tolist()
table_data = [[row[0]] + [f'{v:.2f}' for v in row[1:]] for row in table_data]

table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, len(table_data) + 1):
    color = '#eaf4fb' if i % 2 == 0 else '#ffffff'
    for j in range(len(col_labels)):
        table[i, j].set_facecolor(color)

for j in range(len(col_labels)):
    table[1, j].set_facecolor('#d5f5e3')
    table[1, j].set_text_props(fontweight='bold')

plt.title('Apple Stock - ML Algorithm Performance Comparison', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('apple_metrics_table.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n")
best_acc = metrics_df.iloc[0]
worst_acc = metrics_df.iloc[-1]
best_f1 = pd.DataFrame(metrics_list).sort_values('F1-Score', ascending=False).iloc[0]

print(f"Highest Accuracy: {best_acc['Algorithm']} ({best_acc['Accuracy']:.2f}%)")
print(f"Lowest Accuracy:  {worst_acc['Algorithm']} ({worst_acc['Accuracy']:.2f}%)")
print(f"Best Overall F1-Score: {best_f1['Algorithm']} ({best_f1['F1-Score']:.2f}%)")
