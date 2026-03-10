import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
data_path = "filtered_listings.csv"
data = pd.read_csv(data_path)

# Ensure required columns exist
required_columns = ["id", "name", "host_id", "host_name", "neighbourhood_group", "neighbourhood", "is_fake"]
if not all(col in data.columns for col in required_columns):
    raise ValueError("Filtered listings CSV is missing required columns!")

# Define features and target
X = data.drop(columns=required_columns)
y = data["is_fake"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define hyperparameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}

# Initialize XGBoost and GridSearchCV
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best model
best_xgb = grid_search.best_estimator_

# Evaluate model
y_pred = best_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(best_xgb, "xgboost_model.pkl")
print("Model saved successfully!")
