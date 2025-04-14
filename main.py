from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("ufc-master.csv")

# Drop high-missing and irrelevant columns
drop_cols = ['RedFighter', 'BlueFighter', 'Date', 'Location', 'Country']
df.drop(columns=drop_cols, inplace=True, errors='ignore')
df.dropna(axis=1, thresh=int(0.7 * len(df)), inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
if 'Winner' in categorical_cols:
    categorical_cols = categorical_cols.drop('Winner')
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    df[col] = LabelEncoder().fit_transform(df[col])

# Encode target
df['Winner'] = df['Winner'].map({'Red': 1, 'Blue': 0})

# Feature engineering
df['ReachDiff'] = df['RedReachCms'] - df['BlueReachCms']
df['HeightDiff'] = df['RedHeightCms'] - df['BlueHeightCms']
df['AgeDiff'] = df['RedAge'] - df['BlueAge']

# Standardize numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('Winner')
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Train/test split
X = df.drop('Winner', axis=1)
y = df['Winner']
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Subsample training set to 4k rows stratified
X_train_full_temp = X_train_full.copy()
X_train_full_temp['Winner'] = y_train_full
df_red = X_train_full_temp[X_train_full_temp['Winner'] == 1]
df_blue = X_train_full_temp[X_train_full_temp['Winner'] == 0]
red_needed = int(4000 * (df_red.shape[0] / len(X_train_full_temp)))
blue_needed = 4000 - red_needed
df_red_sample = df_red.sample(n=min(red_needed, df_red.shape[0]), random_state=42)
df_blue_sample = df_blue.sample(n=min(blue_needed, df_blue.shape[0]), random_state=42)
df_train_sampled = pd.concat([df_red_sample, df_blue_sample]).sample(frac=1.0, random_state=42)
y_train = df_train_sampled['Winner']
X_train = df_train_sampled.drop('Winner', axis=1)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
grid_log = GridSearchCV(log_reg, {'C': [0.01, 0.1, 1, 10]}, cv=5, scoring='accuracy', n_jobs=-1)
grid_log.fit(X_train, y_train)
log_preds = grid_log.best_estimator_.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, log_preds))

# Random Forest
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, {'n_estimators': [100], 'max_depth': [10, 20], 'min_samples_split': [5]},
                       cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)
rf_preds = grid_rf.best_estimator_.predict(X_test)
print("Random Forest:")
print(classification_report(y_test, rf_preds))

# XGBoost with GridSearchCV hyperparameter tuning
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
xgb_best = grid_xgb.best_estimator_
xgb_preds = xgb_best.predict(X_test)
print("XGBoost with GridSearchCV:")
print(classification_report(y_test, xgb_preds))
