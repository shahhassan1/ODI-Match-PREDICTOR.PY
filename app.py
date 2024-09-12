

# -------------------------------------------------
# Now for the Streamlit app part
# Create a new file, e.g., `app.py`, for your Streamlit app

import streamlit as st
import pandas as pd
import joblib

# Load the trained model and saved LabelEncoders
model = joblib.load("hassan.joblib")
team1_encode = joblib.load("team1_encoder.joblib")
team2_encode = joblib.load("team2_encoder.joblib")
venue_encode = joblib.load("venue_encoder.joblib")
winner_encode = joblib.load("winner_encoder.joblib")

# Get the original labels used during training (for the dropdowns)
teams = team1_encode.classes_  # Original team names
venues = venue_encode.classes_  # Original venue names

# User inputs via Streamlit
st.title("ODI Match Outcome Predictor")

team1 = st.selectbox('Select Team 1', teams)
team2 = st.selectbox('Select Team 2', teams)
venue = st.selectbox('Select Venue', venues)

# Preprocess inputs by transforming them using the loaded encoders
team1_encoded = team1_encode.transform([team1])
team2_encoded = team2_encode.transform([team2])
venue_encoded = venue_encode.transform([venue])

# Prepare input data for prediction
input_data = pd.DataFrame({'team1': team1_encoded, 'team2': team2_encoded, 'venue': venue_encoded})

# Predict and decode the predicted winner back to its original label
if st.button('Predict Winner'):
    prediction = model.predict(input_data)
    winner = winner_encode.inverse_transform(prediction)
    st.write(f"The predicted winner is: {winner[0]}")
