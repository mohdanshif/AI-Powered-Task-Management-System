from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import base64

# Import your main functions
from main import (
    load_data, preprocess, feature_engineering, prepare_targets,
    build_pipelines, prepare_timeseries, arima_forecast, prophet_forecast,
    workload_balancing, display_accuracy_dashboard
)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# ============================================================
# GLOBAL VARIABLES
# ============================================================
uploaded_df = None
status_model = None
priority_model = None
status_acc = 0
priority_acc = 0
workload_df = None


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_df

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Load CSV to DataFrame
    stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    uploaded_df = pd.read_csv(stream)
    print(f"✅ Uploaded dataset: {uploaded_df.shape[0]} rows")

    return jsonify({"message": "File uploaded successfully", "rows": len(uploaded_df)})


@app.route('/train', methods=['POST'])
def train_model():
    global uploaded_df, status_model, priority_model, status_acc, priority_acc, workload_df

    if uploaded_df is None:
        return jsonify({"error": "Please upload a dataset first"}), 400

    # ====== Preprocess ======
    df = preprocess(uploaded_df)
    df = feature_engineering(df)
    y_status, y_priority = prepare_targets(df)

    # ====== Build Pipelines ======
    status_model, priority_model = build_pipelines()

    # ====== Train Status Model ======
    X_train, X_test, y_train, y_test = train_test_split(df, y_status, test_size=0.2, random_state=42, stratify=y_status)
    status_model.fit(X_train, y_train)
    y_pred = status_model.predict(X_test)
    status_acc = accuracy_score(y_test, y_pred)

    print("\nSTATUS MODEL REPORT")
    print(classification_report(y_test, y_pred, digits=3))

    # ====== Train Priority Model ======
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(priority_model, df, y_priority, cv=cv, scoring='accuracy', n_jobs=-1)
    priority_acc = cv_scores.mean()
    priority_model.fit(df, y_priority)

    # ====== Forecast & Workload Balancing ======
    ts = prepare_timeseries(df)
    forecast_df = prophet_forecast(df) if ts is not None else None
    workload_df = workload_balancing(df, forecast_df=forecast_df)

    # ====== Return Results ======
    results = {
        "status_accuracy": round(status_acc * 100, 2),
        "priority_accuracy": round(priority_acc * 100, 2),
        "workload": workload_df.to_dict(orient="records")
    }

    return jsonify(results)


@app.route('/workload')
def get_workload():
    global workload_df
    if workload_df is None:
        return jsonify({"error": "No workload data available"}), 400

    return workload_df.to_json(orient='records')


@app.route('/accuracy_dashboard')
def accuracy_dashboard():
    global status_acc, priority_acc
    if status_acc == 0 or priority_acc == 0:
        return jsonify({"error": "Model not trained yet"}), 400

    # Create dashboard figure and return as image
    plt.figure(figsize=(6, 4))
    plt.bar(['Status Model', 'Priority Model'], [status_acc * 100, priority_acc * 100])
    plt.title("Model Accuracy Dashboard")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 110)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return f"<img src='data:image/png;base64,{img_base64}'/>"


if __name__ == '__main__':
    app.run(debug=True)
