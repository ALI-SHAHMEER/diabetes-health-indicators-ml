import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ── Load models ───────────────────────────────────────────────────────────────
binary_model = joblib.load("/home/ali/AI/diabetes-health-indicators-ml/models/binary_model.joblib")
multi_model  = joblib.load("/home/ali/AI/diabetes-health-indicators-ml/models/multiclass_model.joblib")
reg_model    = joblib.load("/home/ali/AI/diabetes-health-indicators-ml/models/regression_model.joblib")

# ── Stage label mapping (LabelEncoder encodes alphabetically) ─────────────────
stage_labels = {
    0: "Gestational",
    1: "No Diabetes",
    2: "Pre-Diabetes",
    3: "Type 1",
    4: "Type 2",
}

# ── Categorical encoding (alphabetical order = LabelEncoder default) ──────────
CAT_CLASSES = {
    "gender":            ["Female", "Male", "Other"],
    "ethnicity":         ["Asian", "Black", "Hispanic", "Other", "White"],
    "education_level":   ["Graduate", "Highschool", "No formal", "Postgraduate"],
    "income_level":      ["High", "Low", "Lower-Middle", "Middle", "Upper-Middle"],
    "employment_status": ["Employed", "Retired", "Student", "Unemployed"],
    "smoking_status":    ["Current", "Former", "Never"],
}

def encode_cat(col, value):
    return CAT_CLASSES[col].index(value)


def get_inputs():
    st.sidebar.header("🧬 Patient Input")
    st.sidebar.markdown("---")

    st.sidebar.subheader("👤 Demographics")
    gender    = st.sidebar.selectbox("Gender",          ["Male", "Female", "Other"])
    ethnicity = st.sidebar.selectbox("Ethnicity",       ["Asian", "White", "Hispanic", "Black", "Other"])
    edu       = st.sidebar.selectbox("Education Level", ["Highschool", "Graduate", "Postgraduate", "No formal"])
    income    = st.sidebar.selectbox("Income Level",    ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"])
    employ    = st.sidebar.selectbox("Employment",      ["Employed", "Unemployed", "Retired", "Student"])
    smoking   = st.sidebar.selectbox("Smoking Status",  ["Never", "Former", "Current"])

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
                                format_func=lambda x: "Yes" if x else "No",
                                key="fam")
    hypert   = st.sidebar.radio("Hypertension History", [0, 1],
                                format_func=lambda x: "Yes" if x else "No",
                                key="hyp")
    cardio   = st.sidebar.radio("Cardiovascular History", [0, 1],
                                format_func=lambda x: "Yes" if x else "No",
                                key="card")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Clinical Measurements")
    bmi     = st.sidebar.slider("BMI", 15.0, 50.0, 25.0, 0.1)
    whr     = st.sidebar.slider("Waist-to-Hip Ratio", 0.60, 1.20, 0.85, 0.01)
    sys_bp  = st.sidebar.slider("Systolic BP (mmHg)", 80, 200, 120)
    dia_bp  = st.sidebar.slider("Diastolic BP (mmHg)", 50, 130, 80)
    hr      = st.sidebar.slider("Heart Rate (bpm)", 40, 130, 72)
    chol    = st.sidebar.slider("Total Cholesterol", 100, 400, 200)
    hdl     = st.sidebar.slider("HDL Cholesterol", 20, 100, 55)
    ldl     = st.sidebar.slider("LDL Cholesterol", 40, 250, 120)
    trig    = st.sidebar.slider("Triglycerides", 30, 400, 150)
    gluc_f  = st.sidebar.slider("Fasting Glucose (mg/dL)", 60, 200, 100)
    gluc_p  = st.sidebar.slider("Postprandial Glucose", 80, 350, 140)
    insulin = st.sidebar.slider("Insulin (μIU/mL)", 0.0, 30.0, 5.0, 0.1)
    hba1c   = st.sidebar.slider("HbA1c (%)", 4.0, 10.0, 5.7, 0.1)

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


# ── App ───────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Health Indicators Predictor")

inputs = get_inputs()

tab1, tab2, tab3 = st.tabs([
    "🔵 Binary Classification",
    "🟢 Stage Prediction",
    "📈 Risk Score",
])

with tab1:
    st.header("Will this patient be diagnosed with Diabetes?")
    if st.button("Predict Diagnosis", key="btn_binary"):
        pred  = binary_model.predict([inputs])[0]
        proba = binary_model.predict_proba([inputs])[0]
        if pred == 1:
            st.error(f"⚠️ Diabetic: **YES** — Confidence: {proba[1]*100:.1f}%")
        else:
            st.success(f"✅ Diabetic: **NO** — Confidence: {proba[0]*100:.1f}%")
        col1, col2 = st.columns(2)
        col1.metric("No Diabetes Probability", f"{proba[0]*100:.1f}%")
        col2.metric("Diabetic Probability",    f"{proba[1]*100:.1f}%")

with tab2:
    st.header("What stage of Diabetes?")
    if st.button("Predict Stage", key="btn_stage"):
        pred  = multi_model.predict([inputs])[0]
        label = stage_labels[pred]
        st.success(f"🔬 Predicted Stage: **{label}**")
        if hasattr(multi_model, "predict_proba"):
            probas = multi_model.predict_proba([inputs])[0]
            prob_df = pd.DataFrame({
                "Stage":       [stage_labels[i] for i in range(len(probas))],
                "Probability": probas,
            }).sort_values("Probability", ascending=False).reset_index(drop=True)
            st.dataframe(
                prob_df.style.format({"Probability": "{:.4f}"}),
                use_container_width=True,
            )

with tab3:
    st.header("What is the Diabetes Risk Score?")
    if st.button("Predict Risk Score", key="btn_risk"):
        pred = float(reg_model.predict([inputs])[0])
        st.metric("Risk Score", f"{pred:.2f}")
        st.progress(float(min(pred / 70.0, 1.0)))
        if pred < 20:
            st.success("🟢 Low Risk")
        elif pred < 35:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")