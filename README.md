# **🧠 AI-Powered Task Management System**


A fully optimized AI system that achieves near-perfect accuracy in predicting task priority, forecasting workload, and balancing resources — powered by Machine Learning and NLP.







## **📘 Project Overview**

This AI-Powered Task Management System uses data-driven intelligence to automatically:

Predict task completion status

Classify task priority (High/Medium/Low)

Forecast workload trends using ARIMA and Prophet

Balance workload across users intelligently

Visualize accuracy and load distribution through dashboards

⚡ The system consistently achieves ~99–100% accuracy due to optimized feature engineering, TF-IDF text vectorization, and Random Forest tuning.

## **⚙️ Technologies Used**
Category	Tools / Libraries
Programming	Python 3.10+
ML/NLP	scikit-learn, xgboost, TF-IDF
Forecasting	prophet, statsmodels (ARIMA)
Visualization	matplotlib, seaborn
Data Handling	pandas, numpy
## **📂 Project Structure**
AI-Powered-Task-Management-System/
│
├── project.py                     # Main AI logic (status, priority, forecasting, balancing)
├── synthetic_task_dataset.csv   # Input dataset (sample/synthetic)
└── README.md                    # Project documentation

## **🧩 Core Modules**
### **🗓 Week 1 — Data Loading & Cleaning**

Parses datetime fields and fills missing values

Calculates derived metrics such as duration_minutes

### **💬 Week 2 — NLP + Feature Engineering**

Combines task text fields via TF-IDF

Extracts semantic and temporal features like day of week, hours, title length

### **📈 Week 3 — Forecasting & Modeling**

Status Model: Predicts completion status (Random Forest)

Priority Model: Predicts task urgency (Random Forest / XGBoost)

Forecast Models: Predicts workload trends (ARIMA or Prophet)

### **⚖️ Week 3 — Workload Balancing**

Balances workload dynamically using predicted task loads

Highlights overloaded and underloaded users

### **📊 Week 4 — Dashboard**

Visual summary of model accuracy

Bar chart comparing Status vs Priority model performance

## **🧮 How to Run the Project**
python main.py


Install Dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn prophet xgboost statsmodels

## **📊 Sample Results**

### **1️⃣ Model Accuracy Dashboard**

Model	Accuracy (%)
Task Status Model	99.87%
Priority Model	100.00%

### **2️⃣ Workload Balancing Example**

--- WORKLOAD BALANCING REPORT ---
  user_id  current_load  predicted_future_load  total_expected_load  balance_action
0      U1             8                     12                   20     balanced
1      U2             6                      8                   14     take_more_tasks
2      U3            15                     10                   25     delegate_tasks
3      U4            10                      9                   19     balanced

## **📅 Future Enhancements**

✅ Add dataset import from live task management tools (e.g., Jira, Asana)

✅ Improve explainability using SHAP values

🔜 Implement anomaly detection for overdue tasks

🔜 Automate reallocation recommendations
