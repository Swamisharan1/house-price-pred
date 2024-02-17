import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
import streamlit as st

# Load the data
url = 'https://github.com/Swamisharan1/house-price-pred/blob/main/Housing.csv'  
df = pd.read_csv(url,delimiter='\t')

# One-hot encoding
one_hot = pd.get_dummies(df,columns =['furnishingstatus'],drop_first = True )

# Label encoding
label_encoder = LabelEncoder()
for column in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
    one_hot[column] = label_encoder.fit_transform(one_hot[column])

# Define the base models
model1 = RandomForestRegressor(random_state=0)
model2 = GradientBoostingRegressor(random_state=0)
model3 = LinearRegression()

# Grid search for RandomForestRegressor
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth' : [None, 5, 10],
    'min_samples_split': [2, 5],
}
grid_search_rf = GridSearchCV(estimator=model1, param_grid=param_grid_rf, cv=3, n_jobs=-1)
grid_search_rf.fit(x_train, y_train)
best_params_rf = grid_search_rf.best_params_

# Grid search for GradientBoostingRegressor
param_grid_gb = {
    'n_estimators': [100, 200, 300,400,500],
    'learning_rate' : [0.1, 0.05, 0.01,0.03],
    'max_depth' : [3, 4, 5,7],
}
grid_search_gb = GridSearchCV(estimator=model2, param_grid=param_grid_gb, cv=3, n_jobs=-1)
grid_search_gb.fit(x_train, y_train)
best_params_gb = grid_search_gb.best_params_

# Best models
model1_best = RandomForestRegressor(n_estimators=best_params_rf['n_estimators'], 
                                    max_depth=best_params_rf['max_depth'], 
                                    min_samples_split=best_params_rf['min_samples_split'], 
                                    random_state=0)
model2_best = GradientBoostingRegressor(n_estimators=best_params_gb['n_estimators'], 
                                        learning_rate=best_params_gb['learning_rate'], 
                                        max_depth=best_params_gb['max_depth'], 
                                        random_state=0)

# Ensemble model
ensemble_best = VotingRegressor(estimators=[('rf', model1_best), ('gb', model2_best), ('lr', model3)])
ensemble_best.fit(x_train, y_train)

# Prediction
y_pred_best = ensemble_best.predict(x_test)
r2_best = r2_score(y_test, y_pred_best)
print(f"The R-squared score of the fine-tuned ensemble model on the test set: {r2_best:.4f}")

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
