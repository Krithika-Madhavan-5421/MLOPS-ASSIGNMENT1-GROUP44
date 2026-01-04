import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"  # or Minikube URL

st.title("Heart Disease Risk Prediction")

age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex", [0, 1])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
thalach = st.number_input("Max Heart Rate", 60, 220, 150)

if st.button("Predict"):
    payload = {
        "age": age,
        "sex": sex,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
        st.info(f"Confidence: {result['confidence']}")
    else:
        st.error("API call failed")