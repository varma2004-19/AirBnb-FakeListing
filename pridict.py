import pandas as pd
import xgboost as xgb
import pickle

def load_model():
    """Load the trained XGBoost model."""
    with open("xgboost1_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def preprocess_input_data(file_path):
    """Load and preprocess input data to match trained model features."""
    
    # List of features used during training
    trained_features = [
        'neighbourhood', 'latitude', 'longitude', 'minimum_nights', 
        'reviews_per_month', 'calculated_host_listings_count', 
        'review_year', 'review_month', 'has_reviews', 'host_activity_level', 
        'neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan', 
        'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island', 
        'room_type_Private room', 'room_type_Shared room'
    ]

    # Load dataset
    df = pd.read_csv(file_path)

    # Extract true labels (assuming 'label' column exists)
    if 'is_fake' not in df.columns:
        raise ValueError("Error: The dataset must contain a 'label' column for true values.")

    y_true = df['is_fake']  # True labels (0 = Real, 1 = Fake)
    df = df[trained_features]  # Keep only trained features

    # Convert categorical columns to numerical (if not already encoded)
    if df['host_activity_level'].dtype == 'object':
        df['host_activity_level'] = df['host_activity_level'].astype('category').cat.codes

    return df, y_true

def predict_fake_listings():
    """Load model, preprocess data, make predictions, and display results."""
    
    print("Model loaded successfully!")

    # Load and preprocess input data
    file_path = "filtered_listings.csv"
    print(f"Loading input data from {file_path}...")
    X_input, y_true = preprocess_input_data(file_path)

    # Load trained model
    xgb_model = load_model()

    # Make predictions (1 = Fake, 0 = Real)
    predictions = xgb_model.predict(X_input)

    # Count fakes and reals
    fake_count = sum(predictions == 1)
    real_count = sum(predictions == 0)

    print(f"\nTotal Listings: {len(predictions)}")
    print(f"🟥 Fake Listings: {fake_count}")
    print(f"🟩 Real Listings: {real_count}")

    # Combine true labels and predictions into a DataFrame
    df_results = pd.DataFrame({'True Label': y_true, 'Predicted Label': predictions})

    # Save results
    df_results.to_csv("predictions.csv", index=False)

    print("\nPredictions saved to predictions.csv")
    print("\nSample Predictions:")
    print(df_results.head(10))  # Print first 10 rows

if __name__ == "__main__":
    predict_fake_listings()
