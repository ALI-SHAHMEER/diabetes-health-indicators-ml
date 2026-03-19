import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

<<<<<<< HEAD
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
=======
# ── Load models ───────────────────────────────────────────────────────────────
binary_model = joblib.load('/home/ali/AI/diabetes-health-indicators-ml/models/binary_model.joblib')
multi_model  = joblib.load('/home/ali/AI/diabetes-health-indicators-ml/models/multiclass_model.joblib')
reg_model    = joblib.load('/home/ali/AI/diabetes-health-indicators-ml/models/regression_model.joblib')

# ── Stage labels (LabelEncoder fits alphabetically) ───────────────────────────
# Correct order: Gestational=0, No Diabetes=1, Pre-Diabetes=2, Type 1=3, Type 2=4
stage_labels = {
    0: 'Gestational',
    1: 'No Diabetes',
    2: 'Pre-Diabetes',
    3: 'Type 1',
    4: 'Type 2',
}

# ── Feature columns (must match training order exactly) ───────────────────────
CAT_COLS = ["gender", "ethnicity", "education_level",
            "income_level", "employment_status", "smoking_status"]
NUM_COLS = ["age", "alcohol_consumption_per_week",
            "physical_activity_minutes_per_week", "diet_score",
            "sleep_hours_per_day", "screen_time_hours_per_day",
            "family_history_diabetes", "hypertension_history", "cardiovascular_history",
            "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate",
            "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides",
            "glucose_fasting", "glucose_postprandial", "insulin_level", "hba1c"]

# ── Encode categorical values the same way training did ───────────────────────
# LabelEncoder assigns integer codes alphabetically per column
CAT_CLASSES = {
    "gender":           ["Female", "Male", "Other"],
    "ethnicity":        ["Asian", "Black", "Hispanic", "Other", "White"],
    "education_level":  ["Graduate", "Highschool", "No formal", "Postgraduate"],
    "income_level":     ["High", "Low", "Lower-Middle", "Middle", "Upper-Middle"],
    "employment_status":["Employed", "Retired", "Student", "Unemployed"],
    "smoking_status":   ["Current", "Former", "Never"],
}

def encode_cat(col, value):
    return CAT_CLASSES[col].index(value)

# ── Single input form in sidebar (called once, shared across all tabs) ─────────
def get_inputs():
    st.sidebar.header("🧬 Patient Input")
    st.sidebar.markdown("---")
>>>>>>> fca2819 (Modifying app.py)

    st.sidebar.subheader("👤 Demographics")
    gender   = st.sidebar.selectbox("Gender",           ["Male", "Female", "Other"])
    ethnicity= st.sidebar.selectbox("Ethnicity",        ["Asian", "White", "Hispanic", "Black", "Other"])
    edu      = st.sidebar.selectbox("Education Level",  ["Highschool", "Graduate", "Postgraduate", "No formal"])
    income   = st.sidebar.selectbox("Income Level",     ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"])
    employ   = st.sidebar.selectbox("Employment",       ["Employed", "Unemployed", "Retired", "Student"])
    smoking  = st.sidebar.selectbox("Smoking Status",   ["Never", "Former", "Current"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("🏃 Lifestyle")
    age      = st.sidebar.slider("Age", 18, 90, 45)
    alcohol  = st.sidebar.slider("Alcohol (drinks/week)", 0, 20, 2)
    activity = st.sidebar.slider("Physical Activity (min/week)", 0, 500, 150)
    diet     = st.sidebar.slider("Diet Score (0–10)", 0.0, 10.0, 5.0, 0.1)
    sleep    = st.sidebar.slider("Sleep (hours/day)", 4.0, 12.0, 7.0, 0.1)
    screen   = st.sidebar.slider("Screen Time (hours/day)", 0.0, 16.0, 4.0, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🏥 Medical History")
    fam_hist = st.sidebar.radio("Family History of Diabetes", [0, 1],
                                format_func=lambda x: "Yes" if x else "No")
    hypert   = st.sidebar.radio("Hypertension History", [0, 1],
                                format_func=lambda x: "Yes" if x else "No", key="hyp")
    cardio   = st.sidebar.radio("Cardiovascular History", [0, 1],
                                format_func=lambda x: "Yes" if x else "No", key="card")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Clinical Measurements")
    bmi      = st.sidebar.slider("BMI", 15.0, 50.0, 25.0, 0.1)
    whr      = st.sidebar.slider("Waist-to-Hip Ratio", 0.60, 1.20, 0.85, 0.01)
    sys_bp   = st.sidebar.slider("Systolic BP (mmHg)", 80, 200, 120)
    dia_bp   = st.sidebar.slider("Diastolic BP (mmHg)", 50, 130, 80)
    hr       = st.sidebar.slider("Heart Rate (bpm)", 40, 130, 72)
    chol     = st.sidebar.slider("Total Cholesterol", 100, 400, 200)
    hdl      = st.sidebar.slider("HDL Cholesterol", 20, 100, 55)
    ldl      = st.sidebar.slider("LDL Cholesterol", 40, 250, 120)
    trig     = st.sidebar.slider("Triglycerides", 30, 400, 150)
    gluc_f   = st.sidebar.slider("Fasting Glucose (mg/dL)", 60, 200, 100)
    gluc_p   = st.sidebar.slider("Postprandial Glucose", 80, 350, 140)
    insulin  = st.sidebar.slider("Insulin (μIU/mL)", 0.0, 30.0, 5.0, 0.1)
    hba1c    = st.sidebar.slider("HbA1c (%)", 4.0, 10.0, 5.7, 0.1)

    # Encode categoricals and return full feature vector in training order
    features = [
        encode_cat("gender", gender),
        encode_cat("ethnicity", ethnicity),
        encode_cat("education_level", edu),
        encode_cat("income_level", income),
        encode_cat("employment_status", employ),
        encode_cat("smoking_status", smoking),
        age, alcohol, activity, diet, sleep, screen,
        fam_hist, hypert, cardio,
        bmi, whr, sys_bp, dia_bp, hr,
        chol, hdl, ldl, trig, gluc_f, gluc_p, insulin, hba1c,
    ]
    return features


# ── App Layout ────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Health Indicators Predictor")

# Collect inputs once from sidebar — shared by all three tabs
inputs = get_inputs()

tab1, tab2, tab3 = st.tabs([
    "🔵 Binary Classification",
    "🟢 Stage Prediction",
    "📈 Risk Score",
])

# ── Tab 1: Binary Classification ──────────────────────────────────────────────
with tab1:
    st.header("Will this patient be diagnosed with Diabetes?")
<<<<<<< HEAD
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
=======
    if st.button("Predict Diagnosis", key="btn_binary"):  # unique key required
        pred  = binary_model.predict([inputs])[0]
        proba = binary_model.predict_proba([inputs])[0]
        if pred == 1:
            st.error(f"⚠️ Diabetic: **YES**  (confidence: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ Diabetic: **NO**  (confidence: {proba[0]*100:.1f}%)")
        col1, col2 = st.columns(2)
        col1.metric("No Diabetes Probability", f"{proba[0]*100:.1f}%")
        col2.metric("Diabetic Probability",    f"{proba[1]*100:.1f}%")
>>>>>>> fca2819 (Modifying app.py)

# ── Tab 2: Multiclass Stage ───────────────────────────────────────────────────
with tab2:
    st.header("What stage of Diabetes?")
<<<<<<< HEAD
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
=======
    if st.button("Predict Stage", key="btn_stage"):  # unique key required
        pred  = multi_model.predict([inputs])[0]
        label = stage_labels[pred]
        st.success(f"🔬 Predicted Stage: **{label}**")
        if hasattr(multi_model, "predict_proba"):
            probas = multi_model.predict_proba([inputs])[0]
            prob_df = pd.DataFrame({
                "Stage":       [stage_labels[i] for i in range(len(probas))],
                "Probability": probas,
            }).sort_values("Probability", ascending=False).reset_index(drop=True)
            st.dataframe(prob_df.style.format({"Probability": "{:.4f}"}),
                         use_container_width=True)
>>>>>>> fca2819 (Modifying app.py)

# ── Tab 3: Risk Score Regression ──────────────────────────────────────────────
with tab3:
    st.header("What is the Diabetes Risk Score?")
<<<<<<< HEAD
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
=======
    if st.button("Predict Risk Score", key="btn_risk"):  # unique key required
        pred = reg_model.predict([inputs])[0]
        st.metric("Risk Score", f"{pred:.2f}")
        st.progress(float(min(pred / 70.0, 1.0)))
        if pred < 20:
            st.success("🟢 Low Risk")
        elif pred < 35:
>>>>>>> fca2819 (Modifying app.py)
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")