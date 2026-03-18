import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models
binary_model = joblib.load('models/binary_model.joblib')
multi_model = joblib.load('models/multiclass_model.joblib')
reg_model = joblib.load('models/regression_model.joblib')

stage_labels = {0: 'No Diabetes', 1: 'Pre-Diabetes', 
                2: 'Type 1', 3: 'Type 2', 4: 'Gestational'}

st.title("🩺 Diabetes Health Indicators Predictor")
tab1, tab2, tab3 = st.tabs(["Binary Classification", "Stage Prediction", "Risk Score"])

def get_inputs():
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 90, 45)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        glucose = st.number_input("Fasting Glucose", 50, 300, 100)
        hba1c = st.number_input("HbA1c", 3.0, 15.0, 5.5)
        systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
    with col2:
        diastolic_bp = st.number_input("Diastolic BP", 50, 130, 80)
        cholesterol = st.number_input("Total Cholesterol", 100, 400, 200)
        hdl = st.number_input("HDL Cholesterol", 20, 100, 50)
        ldl = st.number_input("LDL Cholesterol", 50, 300, 100)
        triglycerides = st.number_input("Triglycerides", 50, 500, 150)
    # ... add remaining features
    return [age, bmi, glucose, hba1c, systolic_bp, 
            diastolic_bp, cholesterol, hdl, ldl, triglycerides]

with tab1:
    st.header("Will this patient be diagnosed with Diabetes?")
    inputs = get_inputs()
    if st.button("Predict Diagnosis"):
        pred = binary_model.predict([inputs])[0]
        st.success("✅ Diabetic: YES" if pred == 1 else "✅ Diabetic: NO")

with tab2:
    st.header("What stage of Diabetes?")
    inputs = get_inputs()
    if st.button("Predict Stage"):
        pred = multi_model.predict([inputs])[0]
        st.success(f"🔬 Predicted Stage: {stage_labels[pred]}")

with tab3:
    st.header("What is the Diabetes Risk Score?")
    inputs = get_inputs()
    if st.button("Predict Risk Score"):
        pred = reg_model.predict([inputs])[0]
        st.metric("Risk Score", f"{pred:.2f}")
        st.progress(min(pred / 70, 1.0))
