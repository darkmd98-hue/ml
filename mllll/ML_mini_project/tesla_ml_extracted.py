!pip install imbalanced-learn -q
print('Done!')

====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, ConfusionMatrixDisplay)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier,
                               GradientBoostingClassifier, BaggingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')
print('All libraries imported!')

====

df = pd.read_csv('tesla.csv')
print('Shape:', df.shape)
df.head(10)

====

print('=== Dataset Info ===')
df.info()
print('\n=== Basic Statistics ===')
df.describe()

====

# Check null values
print('=== Null Values Before Replacement ===')
print(df.isnull().sum())
print(f'Total nulls: {df.isnull().sum().sum()}')

====

# Replace any empty strings or whitespace with NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

print('=== Null Values After Replacing Empty Strings with NaN ===')
print(df.isnull().sum())
print(f'Total NaN: {df.isnull().sum().sum()}')

# Visualize missing values
plt.figure(figsize=(8, 4))
sns.heatmap(df.isnull(), cbar=True, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.show()

====

# Parse date and sort
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Target: 1 if next day Close > today's Close, else 0
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Add technical features
df['Price_Change']   = df['Close'] - df['Open']
df['High_Low_Range'] = df['High'] - df['Low']
df['Daily_Return']   = df['Close'].pct_change()
df['MA_5']           = df['Close'].rolling(window=5).mean()
df['MA_10']          = df['Close'].rolling(window=10).mean()
df['Volatility']     = df['Close'].rolling(window=5).std()

# Drop Date column and last row (no target for it)
df.drop(columns=['Date'], inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print('Shape after feature engineering:', df.shape)
df.head()

====

print('=== Class Distribution (Before Balancing) ===')
print(df['Target'].value_counts())
print(f"\n0 = Price Down: {df['Target'].value_counts()[0]}")
print(f"1 = Price Up:   {df['Target'].value_counts()[1]}")

plt.figure(figsize=(6, 4))
df['Target'].value_counts().plot(kind='bar', color=['coral', 'steelblue'], edgecolor='black')
plt.title('Class Distribution Before Balancing')
plt.xlabel('Target (0=Down, 1=Up)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

====

X = df.drop('Target', axis=1)
y = df['Target']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print('=== Class Distribution After SMOTE ===')
print(pd.Series(y_res).value_counts())

plt.figure(figsize=(6, 4))
pd.Series(y_res).value_counts().plot(kind='bar', color=['coral', 'steelblue'], edgecolor='black')
plt.title('Class Distribution After SMOTE Oversampling')
plt.xlabel('Target (0=Down, 1=Up)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

====

# Boxplot for outlier visualization
feature_cols = X_res.columns.tolist()
n = len(feature_cols)
cols_per_row = 4
rows = (n + cols_per_row - 1) // cols_per_row

plt.figure(figsize=(16, rows * 4))
for i, col in enumerate(feature_cols):
    plt.subplot(rows, cols_per_row, i + 1)
    plt.boxplot(X_res[col], patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
    plt.title(col, fontsize=9)
plt.suptitle('Boxplot - Outlier Detection', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

====

# Remove outliers using Z-Score (threshold = 3)
df_combined = pd.DataFrame(X_res, columns=X.columns)
df_combined['Target'] = y_res.values

z_scores = np.abs(stats.zscore(df_combined.select_dtypes(include='number')))
mask = (z_scores < 3).all(axis=1)
df_clean = df_combined[mask].reset_index(drop=True)

print(f'Rows before outlier removal: {len(df_combined)}')
print(f'Rows after outlier removal:  {len(df_clean)}')
print(f'Outliers removed: {len(df_combined) - len(df_clean)}')

====

plt.figure(figsize=(12, 8))
corr = df_clean.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

====

# Feature correlation with target
target_corr = corr['Target'].drop('Target').sort_values(ascending=False)
print('=== Feature Correlation with Target ===')
print(target_corr)

plt.figure(figsize=(10, 5))
target_corr.plot(kind='bar', color='teal', edgecolor='black')
plt.title('Feature Correlation with Target (Up/Down)')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

====

X_final = df_clean.drop('Target', axis=1)
y_final = df_clean['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f'Train: {X_train.shape} | Test: {X_test.shape}')

====

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cm  = confusion_matrix(y_te, y_pred)

    print(f'\n{"="*55}')
    print(f'  {name}')
    print(f'{"="*55}')
    print(f'Accuracy: {acc*100:.2f}%')
    print('\nClassification Report:')
    print(classification_report(y_te, y_pred, target_names=['Down (0)', 'Up (1)']))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.show()

    return acc

results = {}

====

results['Logistic Regression'] = evaluate_model(
    'Logistic Regression',
    LogisticRegression(max_iter=1000, random_state=42),
    X_train_sc, X_test_sc, y_train, y_test
)

====

results['AdaBoost'] = evaluate_model(
    'AdaBoost',
    AdaBoostClassifier(n_estimators=100, random_state=42),
    X_train_sc, X_test_sc, y_train, y_test
)

====

results['KNN'] = evaluate_model(
    'K-Nearest Neighbours (k=5)',
    KNeighborsClassifier(n_neighbors=5),
    X_train_sc, X_test_sc, y_train, y_test
)

====

results['Random Forest'] = evaluate_model(
    'Random Forest',
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train_sc, X_test_sc, y_train, y_test
)

====

results['Decision Tree'] = evaluate_model(
    'Decision Tree',
    DecisionTreeClassifier(random_state=42),
    X_train_sc, X_test_sc, y_train, y_test
)

====

results['SVM'] = evaluate_model(
    'Support Vector Machine',
    SVC(kernel='rbf', random_state=42),
    X_train_sc, X_test_sc, y_train, y_test
)

====

results['Naive Bayes'] = evaluate_model(
    'Naive Bayes',
    GaussianNB(),
    X_train_sc, X_test_sc, y_train, y_test
)

====

results['Gradient Boosting'] = evaluate_model(
    'Gradient Boosting',
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    X_train_sc, X_test_sc, y_train, y_test
)

====

results['Bagging'] = evaluate_model(
    'Bagging Classifier',
    BaggingClassifier(n_estimators=100, random_state=42),
    X_train_sc, X_test_sc, y_train, y_test
)

====

results_df = pd.DataFrame(list(results.items()), columns=['Algorithm', 'Accuracy'])
results_df['Accuracy (%)'] = (results_df['Accuracy'] * 100).round(2)
results_df = results_df.sort_values('Accuracy (%)', ascending=False).reset_index(drop=True)

print('=== Model Accuracy Comparison ===')
print(results_df[['Algorithm', 'Accuracy (%)']].to_string(index=False))

plt.figure(figsize=(12, 6))
bars = plt.bar(results_df['Algorithm'], results_df['Accuracy (%)'],
               color=plt.cm.tab10.colors[:len(results_df)], edgecolor='black')
plt.ylim(0, 115)
plt.ylabel('Accuracy (%)')
plt.title('Tesla Stock - Algorithm Accuracy Comparison')
plt.xticks(rotation=30, ha='right')
for bar, val in zip(bars, results_df['Accuracy (%)']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

best = results_df.iloc[0]
print(f'\nBest Model: {best["Algorithm"]} with {best["Accuracy (%)"]:.2f}% accuracy')

====

from sklearn.metrics import precision_score, recall_score, f1_score

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

metrics_list = []
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    metrics_list.append({
        'Algorithm': name,
        'Accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision': round(precision_score(y_test, y_pred, average='weighted') * 100, 2),
        'Recall':    round(recall_score(y_test, y_pred, average='weighted') * 100, 2),
        'F1-Score':  round(f1_score(y_test, y_pred, average='weighted') * 100, 2)
    })

metrics_df = pd.DataFrame(metrics_list).sort_values('Accuracy', ascending=False).reset_index(drop=True)

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

plt.title('Tesla Stock - ML Algorithm Performance Comparison', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('metrics_table.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved as metrics_table.png')