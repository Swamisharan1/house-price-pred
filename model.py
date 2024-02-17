import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import joblib

# Download the model from GitHub
url = 'https://github.com/Swamisharan1/house-price-pred/blob/main/model.pkl' 
r = requests.get(url)
with open('model.pkl', 'wb') as f:
    f.write(r.content)

# Load the model from the file
ensemble_best = joblib.load('model.pkl')

# Streamlit app
st.sidebar.header('User Input Parameters')

def user_input_features():
    area = st.sidebar.slider('Area', float(df['area'].min()), float(df['area'].max()), float(df['area'].mean()))
    bedrooms = st.sidebar.slider('Bedrooms', int(df['bedrooms'].min()), int(df['bedrooms'].max()), int(df['bedrooms'].mean()))
    bathrooms = st.sidebar.slider('Bathrooms', int(df['bathrooms'].min()), int(df['bathrooms'].max()), int(df['bathrooms'].mean()))
    stories = st.sidebar.slider('Stories', int(df['stories'].min()), int(df['stories'].max()), int(df['stories'].mean()))
    mainroad = st.sidebar.selectbox('Mainroad', options=['yes', 'no'])
    guestroom = st.sidebar.selectbox('Guestroom', options=['yes', 'no'])
    basement = st.sidebar.selectbox('Basement', options=['yes', 'no'])
    hotwaterheating = st.sidebar.selectbox('Hot Water Heating', options=['yes', 'no'])
    airconditioning = st.sidebar.selectbox('Air Conditioning', options=['yes', 'no'])
    parking = st.sidebar.slider('Parking', int(df['parking'].min()), int(df['parking'].max()), int(df['parking'].mean()))
    prefarea = st.sidebar.selectbox('Preferred Area', options=['yes', 'no'])
    furnishingstatus = st.sidebar.selectbox('Furnishing Status', options=['furnished', 'semi-furnished', 'unfurnished'])
    data = {'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms, 'stories': stories, 'mainroad': mainroad, 'guestroom': guestroom, 'basement': basement, 'hotwaterheating': hotwaterheating, 'airconditioning': airconditioning, 'parking': parking, 'prefarea': prefarea, 'furnishingstatus': furnishingstatus}
    return data

df_input = pd.DataFrame(user_input_features(), index=[0])

# Concatenate the input data with the training data
df_combined = pd.concat([df.drop('price', axis=1), df_input], ignore_index=True)

# Perform the same encoding as your training data
label_encoder = LabelEncoder()
for column in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
    df_combined[column] = label_encoder.fit_transform(df_combined[column])

# One-hot encoding for 'furnishingstatus'
df_combined = pd.get_dummies(df_combined, columns=['furnishingstatus'], drop_first=True)

# Separate the input data from the training data
df_input_encoded = df_combined.iloc[-1:]

# Predict and display the output
prediction = ensemble_best.predict(df_input_encoded)
st.write(f"Prediction: {prediction}")
