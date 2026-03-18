import streamlit as st
import joblib
import numpy as np

# Cache models — loaded only once
@st.cache_resource
def load_models():
    binary = joblib.load('models/binary_model.joblib')
    multi  = joblib.load('models/multiclass_model.joblib')
    reg    = joblib.load('models/regression_model.joblib')
    return binary, multi, reg

# Show spinner while loading
with st.spinner("Loading models..."):
    binary_model, multi_model, reg_model = load_models()

st.title("🩺 Diabetes Health Indicators Predictor")
tab1, tab2, tab3 = st.tabs(["Binary", "Stage", "Risk Score"])

stage_labels = {0: 'No Diabetes', 1: 'Pre-Diabetes',
                2: 'Type 1', 3: 'Type 2', 4: 'Gestational'}

with tab1:
    st.header("Will this patient be diagnosed with Diabetes?")
    with st.form("binary_form"):
        col1, col2 = st.columns(2)
        with col1:
            age      = st.slider("Age", 18, 90, 45)
            bmi      = st.number_input("BMI", 10.0, 60.0, 25.0)
            glucose  = st.number_input("Fasting Glucose", 50, 300, 100)
            hba1c    = st.number_input("HbA1c", 3.0, 15.0, 5.5)
            sys_bp   = st.number_input("Systolic BP", 80, 200, 120)
            dia_bp   = st.number_input("Diastolic BP", 50, 130, 80)
            chol     = st.number_input("Total Cholesterol", 100, 400, 200)
        with col2:
            hdl      = st.number_input("HDL Cholesterol", 20, 100, 50)
            ldl      = st.number_input("LDL Cholesterol", 50, 300, 100)
            trig     = st.number_input("Triglycerides", 50, 500, 150)
            insulin  = st.number_input("Insulin Level", 0.0, 300.0, 10.0)
            glu_post = st.number_input("Postprandial Glucose", 50, 400, 140)
            hr       = st.number_input("Heart Rate", 40, 150, 75)
            whr      = st.number_input("Waist-to-Hip Ratio", 0.5, 1.5, 0.85)
        
        submitted = st.form_submit_button("🔍 Predict Diagnosis")
    
    if submitted:
        inputs = [age, bmi, glucose, hba1c, sys_bp, dia_bp,
                  chol, hdl, ldl, trig, insulin, glu_post, hr, whr]
        pred = binary_model.predict([inputs])[0]
        prob = binary_model.predict_proba([inputs])[0][1]
        if pred == 1:
            st.error(f"⚠️ Diabetic: YES  |  Confidence: {prob*100:.1f}%")
        else:
            st.success(f"✅ Diabetic: NO  |  Confidence: {(1-prob)*100:.1f}%")

with tab2:
    st.header("What stage of Diabetes?")
    with st.form("multi_form"):
        col1, col2 = st.columns(2)
        with col1:
            age2     = st.slider("Age", 18, 90, 45)
            bmi2     = st.number_input("BMI", 10.0, 60.0, 25.0)
            glucose2 = st.number_input("Fasting Glucose", 50, 300, 100)
            hba1c2   = st.number_input("HbA1c", 3.0, 15.0, 5.5)
        with col2:
            sys_bp2  = st.number_input("Systolic BP", 80, 200, 120)
            dia_bp2  = st.number_input("Diastolic BP", 50, 130, 80)
            chol2    = st.number_input("Total Cholesterol", 100, 400, 200)
            hdl2     = st.number_input("HDL Cholesterol", 20, 100, 50)
        
        submitted2 = st.form_submit_button("🔬 Predict Stage")
    
    if submitted2:
        inputs2 = [age2, bmi2, glucose2, hba1c2, 
                   sys_bp2, dia_bp2, chol2, hdl2]
        pred2 = multi_model.predict([inputs2])[0]
        st.success(f"📋 Predicted Stage: **{stage_labels[pred2]}**")

with tab3:
    st.header("What is the Diabetes Risk Score?")
    with st.form("reg_form"):
        col1, col2 = st.columns(2)
        with col1:
            age3    = st.slider("Age", 18, 90, 45)
            bmi3    = st.number_input("BMI", 10.0, 60.0, 25.0)
            hba1c3  = st.number_input("HbA1c", 3.0, 15.0, 5.5)
            chol3   = st.number_input("Total Cholesterol", 100, 400, 200)
        with col2:
            glucose3 = st.number_input("Fasting Glucose", 50, 300, 100)
            sys_bp3  = st.number_input("Systolic BP", 80, 200, 120)
            insulin3 = st.number_input("Insulin Level", 0.0, 300.0, 10.0)
            trig3    = st.number_input("Triglycerides", 50, 500, 150)
        
        submitted3 = st.form_submit_button("📈 Predict Risk Score")
    
    if submitted3:
        inputs3 = [age3, bmi3, hba1c3, chol3, 
                   glucose3, sys_bp3, insulin3, trig3]
        pred3 = reg_model.predict([inputs3])[0]
        st.metric("Risk Score", f"{pred3:.2f} / 70")
        st.progress(min(pred3 / 70.0, 1.0))
        if pred3 < 20:
            st.success("🟢 Low Risk")
        elif pred3 < 40:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")