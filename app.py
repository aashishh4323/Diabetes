import streamlit as st
import numpy as np
import joblib

# Load artifacts
model = joblib.load("model/rf_model.pkl")
scaler = joblib.load("model/scaler.pkl")
features = joblib.load("model/features.pkl")

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction System")
st.write("Educational screening tool (not for medical diagnosis)")

# Input fields (must match feature order)
inputs = {}

inputs["Pregnancies"] = st.number_input("Pregnancies", 0, 20, 1)
inputs["Glucose"] = st.number_input("Glucose Level", 0, 300, 120)
inputs["BloodPressure"] = st.number_input("Blood Pressure", 0, 200, 70)
inputs["SkinThickness"] = st.number_input("Skin Thickness", 0, 100, 20)
inputs["Insulin"] = st.number_input("Insulin", 0, 900, 80)
inputs["BMI"] = st.number_input("BMI", 0.0, 70.0, 25.0)
inputs["DiabetesPedigreeFunction"] = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
inputs["Age"] = st.number_input("Age", 1, 100, 30)

if st.button("Predict"):
    X = np.array([[inputs[f] for f in features]])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ High Risk of Diabetes\n\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Low Risk of Diabetes\n\nProbability: {prob:.2f}")

st.caption("⚠️ This tool is for educational purposes only.")

