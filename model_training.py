import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data_path = "filtered_listings.csv"
data = pd.read_csv(data_path)

# Ensure required columns exist
if "is_fake" not in data.columns:
    raise ValueError("Dataset is missing the 'is_fake' column!")

# Drop missing values
data.dropna(inplace=True)
data_numeric = data.copy()
for col in data_numeric.select_dtypes(include=["object"]).columns:
    data_numeric[col] = data_numeric[col].astype("category").cat.codes

# Identify potential data leakage
correlation_matrix = data_numeric.corr()
high_corr_features = correlation_matrix["is_fake"].abs().sort_values(ascending=False)

print("\nFeatures highly correlated with 'is_fake':\n", high_corr_features)

# Drop features that are too strongly correlated (modify if needed)
leakage_features = ["price", "number_of_reviews", "availability_365"]
data = data.drop(columns=[col for col in leakage_features if col in data.columns])

# Define features and target
X = data.drop(columns=["is_fake"])
y = data["is_fake"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Convert categorical columns to 'category' type for XGBoost
for col in categorical_cols:
    X[col] = X[col].astype("category")

# Split dataset (stratify to balance classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print class distributions for train/test
print("Train Set - Fake Listings Count:\n", y_train.value_counts())
print("Test Set - Fake Listings Count:\n", y_test.value_counts())

# Define hyperparameter grid (Reduced for efficiency)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# Initialize XGBoost model
xgb = XGBClassifier(eval_metric="logloss", random_state=42, enable_categorical=True)

# Hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best model
best_xgb = grid_search.best_estimator_

# Evaluate model
y_pred = best_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for XGBoost Model")
plt.show()

# Feature Importance
feature_importances = best_xgb.feature_importances_
if not np.isnan(feature_importances).any():  # Ensure valid values exist
    sorted_idx = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(
    x=feature_importances[sorted_idx][:10], 
    y=np.array(X.columns)[sorted_idx][:10], 
    palette="viridis"
).set(title="Top 10 Feature Importances")

    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importances")
    plt.show()

# Save model
joblib.dump(best_xgb, "xgboost1_model.pkl")
print("\nModel saved successfully!")
