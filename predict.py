import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

# Load the trained model
model_path = "xgboost1_model.pkl"
with open(model_path, "rb") as file:
    xgb_model = pickle.load(file)

print("Model loaded successfully!")

# Function to take user input and make predictions
def predict_manual_input():
    print("\nEnter the listing details for prediction:")
    
    neighbourhood = input("Enter neighbourhood: ")
    latitude = float(input("Enter latitude: "))
    longitude = float(input("Enter longitude: "))
    minimum_nights = int(input("Enter minimum nights: "))
    reviews_per_month = float(input("Enter reviews per month: "))
    calculated_host_listings_count = int(input("Enter host listing count: "))
    review_year = int(input("Enter review year: "))
    review_month = int(input("Enter review month: "))
    has_reviews = int(input("Has reviews? (1 for Yes, 0 for No): "))
    host_activity_level = float(input("Enter host activity level: "))

    # One-hot encoding for categorical fields
    neighbourhood_group = input("Enter neighbourhood group (Brooklyn/Manhattan/Queens/Staten Island): ")
    room_type = input("Enter room type (Private room/Shared room): ")

    # Create input DataFrame
    input_data = {
        "neighbourhood": [neighbourhood],
        "latitude": [latitude],
        "longitude": [longitude],
        "minimum_nights": [minimum_nights],
        "reviews_per_month": [reviews_per_month],
        "calculated_host_listings_count": [calculated_host_listings_count],
        "review_year": [review_year],
        "review_month": [review_month],
        "has_reviews": [has_reviews],
        "host_activity_level": [host_activity_level],
        "neighbourhood_group_Brooklyn": [1 if neighbourhood_group == "Brooklyn" else 0],
        "neighbourhood_group_Manhattan": [1 if neighbourhood_group == "Manhattan" else 0],
        "neighbourhood_group_Queens": [1 if neighbourhood_group == "Queens" else 0],
        "neighbourhood_group_Staten Island": [1 if neighbourhood_group == "Staten Island" else 0],
        "room_type_Private room": [1 if room_type == "Private room" else 0],
        "room_type_Shared room": [1 if room_type == "Shared room" else 0],
    }

    df_input = pd.DataFrame(input_data)

    # Make prediction
    prediction = xgb_model.predict(df_input)[0]

    # Display result
    print("\n🛑 Fake Listing Detected!" if prediction == 1 else "\n✅ Real Listing!")

# Run prediction on user input
predict_manual_input()
