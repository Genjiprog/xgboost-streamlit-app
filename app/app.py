import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing objects
model = joblib.load('model/xgboost_model.pkl')
scaler = joblib.load('model/scaler.pkl')      # MinMaxScaler
encoder = joblib.load('model/encoder.pkl')    # LabelEncoder

# Selected features used in training
selected_features = ['HadAngina', 'HadStroke', 'HadCOPD', 'HadKidneyDisease',
                     'HadArthritis', 'HadDiabetes', 'DeafOrHardOfHearing',
                     'DifficultyWalking', 'ChestScan', 'AgeCategory']

st.title("Heart Attack Risk Prediction")
st.write("Enter the following information to predict the risk of a heart attack using a trained XGBoost model.")

# Helper function
def binary_input(label):
    return st.selectbox(label, ['No', 'Yes']) == 'Yes'

# Input fields
inputs = {
    'HadAngina': binary_input('Has the patient had Angina?'),
    'HadStroke': binary_input('Has the patient had a Stroke?'),
    'HadCOPD': binary_input('Has the patient had COPD?'),
    'HadKidneyDisease': binary_input('Has the patient had Kidney Disease?'),
    'HadArthritis': binary_input('Has the patient had Arthritis?'),
    'HadDiabetes': binary_input('Has the patient had Diabetes?'),
    'DeafOrHardOfHearing': binary_input('Is the patient Deaf or Hard of Hearing?'),
    'DifficultyWalking': binary_input('Does the patient have difficulty walking or climbing stairs?'),
    'ChestScan': binary_input('Has the patient had a Chest CT Scan?'),
    'AgeCategory': st.selectbox('Age Category', sorted([
        '18-24', '25-29', '30-34', '35-39', '40-44',
        '45-49', '50-54', '55-59', '60-64', '65-69',
        '70-74', '75-79', '80+'
    ]))
}

# Convert to DataFrame
input_df = pd.DataFrame([inputs])

# Encode AgeCategory
input_df['AgeCategory'] = encoder.transform(input_df['AgeCategory'])

# Convert boolean to int
for col in input_df.columns:
    if input_df[col].dtype == 'bool':
        input_df[col] = input_df[col].astype(int)

# Scale numerical data
input_scaled = scaler.transform(input_df)

# Predict
threshold = 0.3
proba = model.predict_proba(input_scaled)[:, 1]
prediction = (proba >= threshold).astype(int)[0]

# Output result
st.subheader("Prediction Result")
if prediction == 1:
    st.error("⚠️ High risk of a heart attack.")
else:
    st.success("✅ Low risk of a heart attack.")

st.write(f"Prediction Probability: **{proba[0]:.2f}**")
