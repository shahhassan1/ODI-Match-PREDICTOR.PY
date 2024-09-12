# Complete Code: Training and Streamlit App in One

# First, you need to train the model and save it along with the LabelEncoders

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv('ODI_Match_info.csv')

# Drop missing values (adjust to your dataset's needs)
df = df.dropna(subset=['city', 'umpire3', 'player_of_match'])

# Initialize the LabelEncoders
team1_encode = LabelEncoder()
team2_encode = LabelEncoder()
venue_encode = LabelEncoder()
winner_encode = LabelEncoder()

# Fit the encoders and transform the relevant columns
df['team1'] = team1_encode.fit_transform(df['team1'])
df['team2'] = team2_encode.fit_transform(df['team2'])
df['venue'] = venue_encode.fit_transform(df['venue'])
df['winner'] = winner_encode.fit_transform(df['winner'])

# Define the features (X) and the target (y)
X = df[['team1', 'team2', 'venue']]
y = df['winner']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a DecisionTreeClassifier model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model (optional, just to see the scores)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train Score: {train_score}")
print(f"Test Score: {test_score}")

# Save the trained model and LabelEncoders
joblib.dump(model, 'hassan.joblib')
joblib.dump(team1_encode, 'team1_encoder.joblib')
joblib.dump(team2_encode, 'team2_encoder.joblib')
joblib.dump(venue_encode, 'venue_encoder.joblib')
joblib.dump(winner_encode, 'winner_encoder.joblib')