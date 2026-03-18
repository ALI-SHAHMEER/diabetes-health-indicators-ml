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

# ✅ Pass a unique 'tab' prefix to every widget key
def get_inputs(tab):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 90, 45, key=f"age_{tab}")
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0, key=f"bmi_{tab}")
        glucose = st.number_input("Fasting Glucose", 50, 300, 100, key=f"glucose_{tab}")
        hba1c = st.number_input("HbA1c", 3.0, 15.0, 5.5, key=f"hba1c_{tab}")
        systolic_bp = st.number_input("Systolic BP", 80, 200, 120, key=f"sys_bp_{tab}")
        diastolic_bp = st.number_input("Diastolic BP", 50, 130, 80, key=f"dia_bp_{tab}")
        cholesterol = st.number_input("Total Cholesterol", 100, 400, 200, key=f"chol_{tab}")
    with col2:
        hdl = st.number_input("HDL Cholesterol", 20, 100, 50, key=f"hdl_{tab}")
        ldl = st.number_input("LDL Cholesterol", 50, 300, 100, key=f"ldl_{tab}")
        triglycerides = st.number_input("Triglycerides", 50, 500, 150, key=f"trig_{tab}")
        insulin = st.number_input("Insulin Level", 0.0, 300.0, 10.0, key=f"insulin_{tab}")
        glucose_post = st.number_input("Postprandial Glucose", 50, 400, 140, key=f"glu_post_{tab}")
        heart_rate = st.number_input("Heart Rate", 40, 150, 75, key=f"hr_{tab}")
        waist_hip = st.number_input("Waist-to-Hip Ratio", 0.5, 1.5, 0.85, key=f"whr_{tab}")

    col3, col4 = st.columns(2)
    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"], key=f"gender_{tab}")
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"], key=f"smoke_{tab}")
        education = st.selectbox("Education Level", 
                                  ["Highschool", "Bachelor", "Master", "PhD"], 
                                  key=f"edu_{tab}")
        ethnicity = st.selectbox("Ethnicity", 
                                  ["Asian", "White", "Hispanic", "Black", "Other"], 
                                  key=f"eth_{tab}")
    with col4:
        income = st.selectbox("Income Level", 
                               ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"], 
                               key=f"income_{tab}")
        employment = st.selectbox("Employment Status", 
                                   ["Employed", "Unemployed", "Retired", "Student"], 
                                   key=f"emp_{tab}")
        family_history = st.selectbox("Family History of Diabetes", [0, 1], 
                                       key=f"fam_{tab}")
        hypertension = st.selectbox("Hypertension History", [0, 1], 
                                     key=f"hyper_{tab}")
        cardio = st.selectbox("Cardiovascular History", [0, 1], 
                               key=f"cardio_{tab}")

    col5, col6 = st.columns(2)
    with col5:
        alcohol = st.number_input("Alcohol per Week", 0, 30, 2, key=f"alc_{tab}")
        activity = st.number_input("Physical Activity (mins/week)", 0, 600, 150, key=f"act_{tab}")
        diet = st.number_input("Diet Score", 0.0, 10.0, 5.0, key=f"diet_{tab}")
    with col6:
        sleep = st.number_input("Sleep Hours/Day", 3.0, 12.0, 7.0, key=f"sleep_{tab}")
        screen = st.number_input("Screen Time Hours/Day", 0.0, 16.0, 6.0, key=f"screen_{tab}")

    # Encode categoricals the same way as training
    from sklearn.preprocessing import LabelEncoder
    gender_enc = 1 if gender == "Male" else 0
    smoke_map = {"Never": 2, "Former": 1, "Current": 0}
    edu_map = {"Highschool": 1, "Bachelor": 0, "Master": 2, "PhD": 3}
    eth_map = {"Asian": 0, "White": 4, "Hispanic": 2, "Black": 1, "Other": 3}
    income_map = {"Low": 1, "Lower-Middle": 2, "Middle": 3, "Upper-Middle": 4, "High": 0}
    emp_map = {"Employed": 0, "Unemployed": 3, "Retired": 2, "Student": 1}

    return [age, gender_enc, eth_map[ethnicity], edu_map[education],
            income_map[income], emp_map[employment], smoke_map[smoking],
            alcohol, activity, diet, sleep, screen,
            family_history, hypertension, cardio,
            bmi, waist_hip, systolic_bp, diastolic_bp, heart_rate,
            cholesterol, hdl, ldl, triglycerides,
            glucose, glucose_post, insulin, hba1c]

# --- Tab 1: Binary Classification ---
with tab1:
    st.header("🔍 Will this patient be diagnosed with Diabetes?")
    inputs = get_inputs("binary")
    if st.button("Predict Diagnosis", key="btn_binary"):
        pred = binary_model.predict([inputs])[0]
        prob = binary_model.predict_proba([inputs])[0][1]
        if pred == 1:
            st.error(f"⚠️ Diabetic: YES  |  Confidence: {prob*100:.1f}%")
        else:
            st.success(f"✅ Diabetic: NO  |  Confidence: {(1-prob)*100:.1f}%")

# --- Tab 2: Multiclass Classification ---
with tab2:
    st.header("🔬 What stage of Diabetes?")
    inputs = get_inputs("multi")
    if st.button("Predict Stage", key="btn_multi"):
        pred = multi_model.predict([inputs])[0]
        st.success(f"📋 Predicted Stage: **{stage_labels[pred]}**")

# --- Tab 3: Regression ---
with tab3:
    st.header("📈 What is the Diabetes Risk Score?")
    inputs = get_inputs("reg")
    if st.button("Predict Risk Score", key="btn_reg"):
        pred = reg_model.predict([inputs])[0]
        st.metric("Risk Score", f"{pred:.2f} / 70")
        st.progress(min(pred / 70.0, 1.0))
        if pred < 20:
            st.success("🟢 Low Risk")
        elif pred < 40:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")