import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("xgboost_model.pkl")

def predict_with_threshold(model, X, threshold=0.3):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    return pred

st.title("Disease Prediction App")
st.write("Upload CSV untuk prediksi klasifikasi penyakit.")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data:")
    st.dataframe(df)

    prediction = predict_with_threshold(model, df, threshold=0.3)
    df["Prediction"] = prediction
    st.subheader("Hasil Prediksi:")
    st.dataframe(df)
