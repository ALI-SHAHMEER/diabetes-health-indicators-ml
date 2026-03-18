import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_models():
    binary  = joblib.load('models/binary_model.joblib')
    multi   = joblib.load('models/multiclass_model.joblib')
    reg     = joblib.load('models/regression_model.joblib')
    encoder = joblib.load('models/stage_encoder.joblib')
    features = joblib.load('models/feature_cols.joblib')
    return binary, multi, reg, encoder, features

binary_model, multi_model, reg_model, stage_encoder, feature_cols = load_models()

# ── Verify on startup ──────────────────────────────
st.sidebar.write("✅ Binary classes:", binary_model.classes_.tolist())
st.sidebar.write("✅ Features expected:", binary_model.n_features_in_)

st.title("🩺 Diabetes Health Indicators Predictor")
tab1, tab2, tab3 = st.tabs(["Binary", "Stage", "Risk Score"])

def get_inputs(tab):
    col1, col2 = st.columns(2)
    with col1:
        age     = st.slider("Age", 18, 90, 45,          key=f"age_{tab}")
        bmi     = st.number_input("BMI", 10.0, 60.0, 25.0,      key=f"bmi_{tab}")
        glucose = st.number_input("Fasting Glucose", 50, 300, 95, key=f"gluc_{tab}")
        hba1c   = st.number_input("HbA1c", 3.0, 15.0, 5.4,      key=f"hba1c_{tab}")
        sys_bp  = st.number_input("Systolic BP", 80, 200, 120,   key=f"sys_{tab}")
        dia_bp  = st.number_input("Diastolic BP", 50, 130, 80,   key=f"dia_{tab}")
        chol    = st.number_input("Total Cholesterol", 100, 400, 190, key=f"chol_{tab}")
    with col2:
        hdl     = st.number_input("HDL Cholesterol", 20, 100, 55,  key=f"hdl_{tab}")
        ldl     = st.number_input("LDL Cholesterol", 50, 300, 110, key=f"ldl_{tab}")
        trig    = st.number_input("Triglycerides", 50, 500, 140,   key=f"trig_{tab}")
        insulin = st.number_input("Insulin Level", 0.0, 300.0, 8.0, key=f"ins_{tab}")
        glu_post= st.number_input("Postprandial Glucose", 50, 400, 130, key=f"gpost_{tab}")
        hr      = st.number_input("Heart Rate", 40, 150, 72,       key=f"hr_{tab}")
        whr     = st.number_input("Waist-Hip Ratio", 0.5, 1.5, 0.85, key=f"whr_{tab}")

    col3, col4 = st.columns(2)
    with col3:
        gender     = st.selectbox("Gender", ["Female", "Male"],  key=f"gen_{tab}")
        smoking    = st.selectbox("Smoking", ["Current", "Former", "Never"], key=f"smk_{tab}")
        education  = st.selectbox("Education", ["Bachelor", "Highschool", "Master", "PhD"], key=f"edu_{tab}")
        ethnicity  = st.selectbox("Ethnicity", ["Asian", "Black", "Hispanic", "Other", "White"], key=f"eth_{tab}")
    with col4:
        income     = st.selectbox("Income", ["High", "Low", "Lower-Middle", "Middle", "Upper-Middle"], key=f"inc_{tab}")
        employment = st.selectbox("Employment", ["Employed", "Retired", "Student", "Unemployed"], key=f"emp_{tab}")
        family_h   = st.selectbox("Family History Diabetes", [0, 1], key=f"fam_{tab}")
        hyper      = st.selectbox("Hypertension History", [0, 1],    key=f"hyp_{tab}")
        cardio     = st.selectbox("Cardiovascular History", [0, 1],  key=f"car_{tab}")

    col5, col6 = st.columns(2)
    with col5:
        alcohol  = st.number_input("Alcohol/Week", 0, 30, 1,     key=f"alc_{tab}")
        activity = st.number_input("Activity mins/week", 0, 600, 150, key=f"act_{tab}")
        diet     = st.number_input("Diet Score", 0.0, 10.0, 5.0, key=f"diet_{tab}")
    with col6:
        sleep    = st.number_input("Sleep hrs/day", 3.0, 12.0, 7.0,  key=f"slp_{tab}")
        screen   = st.number_input("Screen hrs/day", 0.0, 16.0, 5.0, key=f"scr_{tab}")

    # ── Encode exactly as training ─────────────────
    gender_enc     = ["Female", "Male"].index(gender)
    smoking_enc    = ["Current", "Former", "Never"].index(smoking)
    education_enc  = ["Bachelor", "Highschool", "Master", "PhD"].index(education)
    ethnicity_enc  = ["Asian", "Black", "Hispanic", "Other", "White"].index(ethnicity)
    income_enc     = ["High", "Low", "Lower-Middle", "Middle", "Upper-Middle"].index(income)
    employment_enc = ["Employed", "Retired", "Student", "Unemployed"].index(employment)

    # ── Build input in EXACT feature order ─────────
    input_dict = {
        'age': age, 'gender': gender_enc,
        'ethnicity': ethnicity_enc, 'education_level': education_enc,
        'income_level': income_enc, 'employment_status': employment_enc,
        'smoking_status': smoking_enc,
        'alcohol_consumption_per_week': alcohol,
        'physical_activity_minutes_per_week': activity,
        'diet_score': diet, 'sleep_hours_per_day': sleep,
        'screen_time_hours_per_day': screen,
        'family_history_diabetes': family_h,
        'hypertension_history': hyper,
        'cardiovascular_history': cardio,
        'bmi': bmi, 'waist_to_hip_ratio': whr,
        'systolic_bp': sys_bp, 'diastolic_bp': dia_bp,
        'heart_rate': hr, 'cholesterol_total': chol,
        'hdl_cholesterol': hdl, 'ldl_cholesterol': ldl,
        'triglycerides': trig, 'glucose_fasting': glucose,
        'glucose_postprandial': glu_post,
        'insulin_level': insulin, 'hba1c': hba1c
    }

    # Return as DataFrame with correct column order
    return pd.DataFrame([input_dict])[feature_cols]

# ── Tab 1: Binary ──────────────────────────────────
with tab1:
    st.header("Will this patient be Diabetic?")
    with st.form("form_binary"):
        inputs = get_inputs("binary")
        submitted = st.form_submit_button("🔍 Predict")
    if submitted:
        pred = binary_model.predict(inputs)[0]
        prob = binary_model.predict_proba(inputs)[0][1]
        if pred == 1:
            st.error(f"⚠️ Diabetic: YES  |  Confidence: {prob*100:.1f}%")
        else:
            st.success(f"✅ Diabetic: NO  |  Confidence: {(1-prob)*100:.1f}%")

# ── Tab 2: Multiclass ──────────────────────────────
with tab2:
    st.header("What Diabetes Stage?")
    with st.form("form_multi"):
        inputs = get_inputs("multi")
        submitted2 = st.form_submit_button("🔬 Predict Stage")
    if submitted2:
        pred2 = multi_model.predict(inputs)[0]
        stage = stage_encoder.classes_[pred2]
        st.success(f"📋 Predicted Stage: **{stage}**")

# ── Tab 3: Regression ──────────────────────────────
with tab3:
    st.header("What is the Risk Score?")
    with st.form("form_reg"):
        inputs = get_inputs("reg")
        submitted3 = st.form_submit_button("📈 Predict Score")
    if submitted3:
        score = reg_model.predict(inputs)[0]
        st.metric("Risk Score", f"{score:.2f} / 70")
        st.progress(min(score / 70.0, 1.0))
        if score < 20:
            st.success("🟢 Low Risk")
        elif score < 40:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")