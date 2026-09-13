import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# --------------------------------------------------
# Load saved model
# --------------------------------------------------




BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "airbnb_price_model.pkl"
MODEL_INFO_PATH = BASE_DIR / "airbnb_model_info.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_model_info():
    return joblib.load(MODEL_INFO_PATH)

model = load_model()
model_info = load_model_info()


# --------------------------------------------------
# Get categories from the trained encoder
# --------------------------------------------------

preprocessor = model.named_steps["preprocessor"]

encoder = preprocessor.named_transformers_["cat"]

categories = encoder.categories_

neighbourhood_group_options = list(categories[0])
neighbourhood_options = list(categories[1])
room_type_options = list(categories[2])


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏠 Airbnb Price Predictor")

st.write(
    "Enter the details of an Airbnb listing to estimate "
    "its nightly price."
)

st.divider()


# --------------------------------------------------
# User inputs
# --------------------------------------------------

st.subheader("Listing Information")


neighbourhood_group = st.selectbox(
    "Neighbourhood Group",
    neighbourhood_group_options
)


neighbourhood = st.selectbox(
    "Neighbourhood",
    neighbourhood_options
)


room_type = st.selectbox(
    "Room Type",
    room_type_options
)


latitude = st.number_input(
    "Latitude",
    min_value=40.4,
    max_value=40.95,
    value=40.75,
    format="%.6f"
)


longitude = st.number_input(
    "Longitude",
    min_value=-74.3,
    max_value=-73.6,
    value=-73.98,
    format="%.6f"
)


minimum_nights = st.number_input(
    "Minimum Nights",
    min_value=1,
    max_value=1000,
    value=2,
    step=1
)


number_of_reviews = st.number_input(
    "Number of Reviews",
    min_value=0,
    max_value=1000,
    value=10,
    step=1
)


reviews_per_month = st.number_input(
    "Reviews per Month",
    min_value=0.0,
    max_value=100.0,
    value=1.0,
    step=0.1
)


calculated_host_listings_count = st.number_input(
    "Host Listing Count",
    min_value=1,
    max_value=1000,
    value=1,
    step=1
)


availability_365 = st.number_input(
    "Availability (days/year)",
    min_value=0,
    max_value=365,
    value=200,
    step=1
)


# --------------------------------------------------
# Last review
# --------------------------------------------------

st.subheader("Review Information")

no_review = st.checkbox(
    "No previous review"
)

if no_review:

    days_since_last_review = -1

else:

    last_review = st.date_input(
        "Last Review Date",
        value=pd.Timestamp("2019-06-01"),
        min_value=pd.Timestamp("2011-01-01"),
        max_value=pd.Timestamp("2019-12-31")
    )

    reference_date = pd.Timestamp("2019-12-31")

    days_since_last_review = (
        reference_date -
        pd.Timestamp(last_review)
    ).days


# --------------------------------------------------
# Create input DataFrame
# --------------------------------------------------

input_data = pd.DataFrame({
    "neighbourhood_group": [neighbourhood_group],
    "neighbourhood": [neighbourhood],
    "latitude": [latitude],
    "longitude": [longitude],
    "room_type": [room_type],
    "minimum_nights": [minimum_nights],
    "number_of_reviews": [number_of_reviews],
    "reviews_per_month": [reviews_per_month],
    "calculated_host_listings_count": [
        calculated_host_listings_count
    ],
    "availability_365": [availability_365],
    "days_since_last_review": [
        days_since_last_review
    ]
})


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button(
    "Predict Airbnb Price",
    type="primary",
    use_container_width=True
):

    # Model predicts log(price)
    predicted_log_price = model.predict(input_data)

    # Convert log price back to dollars
    predicted_price = np.expm1(
        predicted_log_price[0]
    )

    st.success(
        f"Estimated Nightly Price: ${predicted_price:,.2f}"
    )

    st.caption(
        "Prediction generated using the trained "
        "Tuned XGBoost model."
    )
