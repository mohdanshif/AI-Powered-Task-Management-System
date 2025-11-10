
"""
Project 1 — AI-Powered Task Management System (Final Full Version)
Includes: Priority Prediction, Forecasting, Workload Balancing & Accuracy Dashboard
Technologies: NLP (TF-IDF), Automation, Time Series Forecasting, ML
"""

import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Optional imports
try:
    import xgboost as xgb
    xgboost_available = True
except Exception:
    xgboost_available = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    statsmodels_available = True
except Exception:
    statsmodels_available = False

try:
    from prophet import Prophet
    prophet_available = True
except Exception:
    prophet_available = False


# ---------- PATH HANDLING ----------
DEFAULT_PATHS = [
    Path('/mnt/data/synthetic_task_dataset.csv'),
    Path('./synthetic_task_dataset.csv'),
]
csv_candidates = list(Path('.').glob('*.csv'))
if csv_candidates:
    DEFAULT_PATHS.extend(csv_candidates)


def find_data_path():
    for p in DEFAULT_PATHS:
        if p.exists():
            return p
    print("⚠️ No CSV found — waiting for upload or Flask run.")
    return None


DATA_PATH = find_data_path()


# ---------- WEEK 1: LOAD & CLEAN ----------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess(df):
    df = df.copy()
    for c in ['created_at', 'due_date', 'completed_at']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')

    if 'status' not in df.columns:
        df['status'] = np.where(df['completed_at'].notna(), 'completed', 'open')

    if 'duration_minutes' not in df.columns and {'created_at', 'completed_at'}.issubset(df.columns):
        df['duration_minutes'] = (df['completed_at'] - df['created_at']).dt.total_seconds() / 60

    df['priority'] = df.get('priority', 'medium').fillna('medium')

    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].fillna('unknown')

    return df


# ---------- WEEK 2: NLP + FEATURES ----------
def build_text_features(df, text_cols=['title', 'description'], max_features=2000):
    existing = [c for c in text_cols if c in df.columns]
    if not existing:
        for c in text_cols:
            df[c] = ''
        existing = text_cols
    df['__text_combined'] = df[existing].astype(str).agg(' '.join, axis=1)
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_text = tfidf.fit_transform(df['__text_combined'])
    print(f"TF-IDF features shape: {X_text.shape}")
    return tfidf, X_text


def feature_engineering(df):
    df = df.copy()

    if 'created_at' in df.columns:
        df['created_dow'] = df['created_at'].dt.dayofweek
        df['created_hour'] = df['created_at'].dt.hour
    else:
        df['created_dow'], df['created_hour'] = -1, -1

    if 'title' in df.columns:
        df['title_len'] = df['title'].astype(str).apply(len)
    else:
        df['title_len'] = 0

    if {'due_date', 'created_at'}.issubset(df.columns):
        df['days_to_due'] = (df['due_date'] - df['created_at']).dt.days
    else:
        df['days_to_due'] = 0

    df['priority_ord'] = df['priority'].map({'low': 0, 'medium': 1, 'high': 2}).fillna(1).astype(int)

    return df


# ---------- WEEK 3: FORECASTING ----------
def prepare_timeseries(df):
    if 'created_at' not in df.columns:
        return None
    ts = df.set_index('created_at').resample('D').size().rename('task_count')
    return ts.asfreq('D').fillna(0)


def arima_forecast(ts, order=(5, 1, 0), steps=30):
    if not statsmodels_available:
        print("statsmodels not installed — skipping ARIMA")
        return None
    model = ARIMA(ts, order=order)
    fit = model.fit()
    forecast_df = fit.get_forecast(steps=steps).summary_frame()
    print("\n--- ARIMA FORECAST SAMPLE ---")
    print(forecast_df.head())
    return forecast_df


def prophet_forecast(df, periods=30):
    if not prophet_available:
        print("prophet not installed — skipping Prophet")
        return None
    daily = df.copy()
    daily['ds'] = pd.to_datetime(daily['created_at']).dt.date
    daily = daily.groupby('ds').size().reset_index(name='y')
    daily['ds'] = pd.to_datetime(daily['ds'])
    m = Prophet()
    m.fit(daily)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    print("\n--- PROPHET FORECAST SAMPLE ---")
    print(forecast[['ds', 'yhat']].tail())
    return forecast


# ---------- WEEK 3: PRIORITY MODEL + WORKLOAD BALANCING ----------
def prepare_ml_data(df, tfidf=None, X_text=None):
    df['target_completed'] = (df['status'] == 'completed').astype(int)
    df['priority_target'] = df['priority'].map({'low': 0, 'medium': 1, 'high': 2}).fillna(1).astype(int)

    num_cols = ['title_len', 'created_dow', 'created_hour', 'days_to_due', 'duration_minutes', 'priority_ord']
    num_cols = [c for c in num_cols if c in df.columns]
    X_num = df[num_cols].fillna(0).astype(float)

    if tfidf is not None and X_text is not None:
        from scipy.sparse import hstack
        X = hstack([X_text, X_num.values])
    else:
        X = X_num.values

    return X, df['target_completed'].values, df['priority_target'].values, num_cols


def train_status_classifier(X, y):
    print("Training Status Classifier (RandomForest)...")
    model = RandomForestClassifier(random_state=42)
    params = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
    gs = GridSearchCV(model, params, cv=3, scoring='accuracy')
    gs.fit(X, y)
    print("Best model params:", gs.best_params_)
    return gs.best_estimator_


def train_priority_model(X, y):
    print("Training Priority Model (RandomForest/XGBoost)...")
    rf = RandomForestClassifier(random_state=42)
    params = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
    gs = GridSearchCV(rf, params, cv=3, scoring='accuracy')
    gs.fit(X, y)
    best_rf = gs.best_estimator_

    if xgboost_available and len(np.unique(y)) > 1:
        xg = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        xg.fit(X, y)
        return best_rf, xg
    return best_rf, None


# ---------- NEW: WORKLOAD BALANCING ----------
def workload_balancing(df, forecast_df=None):
    """
    Balances workload among users using predicted priorities and forecasted workload.
    """
    if 'user_id' not in df.columns:
        df['user_id'] = np.random.choice(['U1', 'U2', 'U3', 'U4'], size=len(df))

    workload = (
        df.groupby('user_id')
        .apply(lambda x: ((x['status'] != 'completed') | (x['priority_ord'] == 2)).sum())
        .reset_index(name='current_load')
    )

    if forecast_df is not None and 'yhat' in forecast_df.columns:
        next_7_days = forecast_df.tail(7)
        future_mean = int(next_7_days['yhat'].mean())
        workload['predicted_future_load'] = np.random.randint(future_mean // 2, future_mean, len(workload))
    else:
        workload['predicted_future_load'] = np.random.randint(5, 15, len(workload))

    workload['total_expected_load'] = workload['current_load'] + workload['predicted_future_load']

    avg_load = workload['total_expected_load'].mean()
    workload['balance_action'] = workload['total_expected_load'].apply(
        lambda x: 'delegate_tasks' if x > avg_load * 1.2 else ('take_more_tasks' if x < avg_load * 0.8 else 'balanced')
    )

    print("\n--- WORKLOAD BALANCING REPORT ---")
    print(workload)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=workload, x='user_id', y='total_expected_load', hue='balance_action')
    plt.title('Workload Balancing Overview')
    plt.ylabel('Total Expected Load')
    plt.xlabel('User')
    plt.tight_layout()
    plt.show()

    return workload


# ---------- WEEK 4: DASHBOARD ----------
def display_accuracy_dashboard(status_acc, priority_cv_mean):
    print("\n========= MODEL ACCURACY DASHBOARD =========")
    print(f"Task Completion (Status) Model Accuracy : {status_acc * 100:.2f}%")
    print(f"Priority Prediction Model CV Mean        : {priority_cv_mean * 100:.2f}%")

    plt.figure(figsize=(6, 4))
    plt.bar(['Status Model', 'Priority Model'], [status_acc * 100, priority_cv_mean * 100], color=['#4CAF50', '#2196F3'])
    plt.ylabel("Accuracy (%)")
    plt.title("AI-Powered Task Management System — Model Performance")
    for i, v in enumerate([status_acc * 100, priority_cv_mean * 100]):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=11, fontweight='bold')
    plt.ylim(0, 110)
    plt.show()


# ---------- MAIN ----------
def main():
    if DATA_PATH is None:
        print("No dataset found — please upload or specify path.")
        return

    df = load_data()
    df = preprocess(df)
    df = feature_engineering(df)
    tfidf, X_text = build_text_features(df)
    X, y_status, y_priority, num_cols = prepare_ml_data(df, tfidf, X_text)

    # Status Model
    X_train, X_test, y_train, y_test = train_test_split(X, y_status, test_size=0.2, random_state=42)
    status_clf = train_status_classifier(X_train, y_train)
    y_pred = status_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n--- STATUS MODEL ---")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # Priority Model
    rf_model, _ = train_priority_model(X, y_priority)
    cv_scores = cross_val_score(rf_model, X, y_priority, cv=3, scoring='accuracy')
    print("\n--- PRIORITY MODEL ---")
    print("CV Scores:", cv_scores, "Mean:", cv_scores.mean())

    # Forecasting
    ts = prepare_timeseries(df)
    prop = None
    if ts is not None and statsmodels_available:
        arima_forecast(ts)
    if prophet_available:
        prop = prophet_forecast(df)

    # Workload Balancing
    workload_balancing(df, forecast_df=prop if prophet_available else None)

    # Dashboard
    display_accuracy_dashboard(acc, cv_scores.mean())


if __name__ == "__main__":
    main()

