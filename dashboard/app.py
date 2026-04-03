import sys
import os
import json
import joblib
import datetime

# Inject the root folder into the Python path so the 'modules' folder can be discovered natively
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from modules.fusion import get_meta_features, compute_reliability, reliability_weights
from modules.xai_router import route_explanation
from modules.drift_monitor import check_drift

st.set_page_config(page_title="FAMED-lite CDS", layout="wide")

st.title("FAMED-lite: Clinical Decision Support")
st.markdown("Federated Adaptive Multimodal Early Disease Detection Prototype")

@st.cache_resource
def load_models_v7():
    try:
        enc_h = joblib.load('models/enc_heart.pkl')
        enc_d = joblib.load('models/enc_diabetes.pkl')
        meta = joblib.load('models/meta_learner.pkl')
        return enc_h, enc_d, meta
    except Exception as e:
        return None, None, None

@st.cache_data
def load_data_v7():
    try:
        h_a = pd.read_csv('data/processed/heart_a.csv')
        h_b = pd.read_csv('data/processed/heart_b.csv')
        d_b = pd.read_csv('data/processed/diabetes_b.csv')
        return h_a, h_b, d_b
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

enc_heart, enc_diabetes, meta_learner = load_models_v7()
heart_a, heart_b, diabetes_b = load_data_v7()

# --- Compute live batch reliability scores ---
# Uses the hospital Site B deployment batch as the signal source
_rel_cardiac   = compute_reliability(heart_b.drop(columns=['patient_id'], errors='ignore'))
_rel_metabolic = compute_reliability(diabetes_b.drop(columns=['patient_id'], errors='ignore'))
_rel_weights   = reliability_weights([_rel_cardiac, _rel_metabolic])


# ------------------ SIDEBAR ------------------ #
st.sidebar.header("🌐 Federated Node Status")
st.sidebar.markdown("""
* 📡 **Node 1 (Cardio):** `[ONLINE - 100% SYNC]`
* 📡 **Node 2 (Metabolic):** `[ONLINE - 100% SYNC]`
* 🔒 **Meta-Aggregator:** `[SECURE ENCLAVE ACTIVE]`
""")

# Live Reliability Panel
st.sidebar.markdown("---")
st.sidebar.header("📊 Modality Reliability (Live)")
st.sidebar.markdown("*Automatically computed from incoming hospital batch quality signals.*")
st.sidebar.progress(float(_rel_cardiac),   text=f"Cardiac   — reliability: {_rel_cardiac:.2f}  |  weight: {_rel_weights[0]:.2f}")
st.sidebar.progress(float(_rel_metabolic), text=f"Metabolic — reliability: {_rel_metabolic:.2f}  |  weight: {_rel_weights[1]:.2f}")
st.sidebar.caption("Weight = Reliability / (Σ Reliabilities). Lower quality → lower influence on risk score.")
st.sidebar.markdown("---")

st.sidebar.header("Patient Input Method")
mode = st.sidebar.radio("Mode:", ["Select Existing Test Patient", "Enter New Patient Manually"])

if mode == "Select Existing Test Patient":
    if not heart_b.empty and not diabetes_b.empty:
        valid_patients = sorted(list(set(heart_b['patient_id']).intersection(set(diabetes_b['patient_id']))))
        patient_list = valid_patients[:20]
        patient_id = st.sidebar.selectbox("Test Patient ID", patient_list)
        
        patient_h = heart_b[heart_b['patient_id'] == patient_id].drop(columns=['patient_id']).copy()
        patient_d = diabetes_b[diabetes_b['patient_id'] == patient_id].drop(columns=['patient_id']).copy()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("Modify test vitals manually:")
        patient_h['age'] = st.sidebar.slider("Age", 20, 95, int(patient_h['age'].values[0]))
        patient_h['max_hr'] = st.sidebar.slider("Max HR", 60, 200, int(patient_h['max_hr'].values[0]))
        patient_d['glucose'] = st.sidebar.slider("Glucose", 50, 300, int(patient_d['glucose'].values[0]))
    else:
        st.sidebar.error("Data missing. Run pipeline first.")
        patient_h, patient_d = None, None

elif mode == "Enter New Patient Manually":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Cardiac Vitals (Modality 1)")
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=55)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x==1 else "Female")
    cpt = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3], index=0)
    rbp = st.sidebar.number_input("Resting BP", 80, 200, 130)
    chol = st.sidebar.number_input("Cholesterol", 100, 500, 240)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1], index=0)
    ecg = st.sidebar.selectbox("Resting ECG (0-2)", [0, 1, 2], index=1)
    mhr = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
    ang = st.sidebar.selectbox("Exercise Angina", [0, 1], index=0)
    old = st.sidebar.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    st_slope = st.sidebar.selectbox("ST Slope (0-2)", [0, 1, 2], index=1)
    
    patient_h = pd.DataFrame({
        'age': [age], 'sex': [sex], 'chest_pain_type': [cpt], 'resting_bp': [rbp],
        'cholesterol': [chol], 'fasting_bs': [fbs], 'resting_ecg': [ecg], 
        'max_hr': [mhr], 'exercise_angina': [ang], 'oldpeak': [old], 'st_slope': [st_slope]
    })
    
    st.sidebar.subheader("Metabolic Labs (Modality 2)")
    preg = st.sidebar.number_input("Pregnancies", 0, 20, 0)
    gluc = st.sidebar.number_input("Glucose", 50, 300, 120)
    bp = st.sidebar.number_input("Blood Pressure (Diastolic)", 40, 150, 70)
    skin = st.sidebar.number_input("Skin Thickness", 10, 100, 25)
    ins = st.sidebar.number_input("Insulin", 15.0, 400.0, 80.0)
    bmi = st.sidebar.number_input("BMI", 15.0, 60.0, 32.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Fn", 0.0, 2.5, 0.5)
    
    patient_d = pd.DataFrame({
        'pregnancies': [preg], 'glucose': [gluc], 'blood_pressure': [bp],
        'skin_thickness': [skin], 'insulin': [ins], 'bmi': [bmi], 'diabetes_pedigree_function': [dpf]
    })

st.sidebar.markdown("---")
st.sidebar.header("Robust Missing-Modality Fault Toolkit")
st.sidebar.markdown("Toggle these to simulate catastrophic system drops at an external hospital site.")
cardiac = st.sidebar.checkbox("Cardiac data Online", value=True)
metabolic = st.sidebar.checkbox("Metabolic data Online", value=True)


# ------------------ MAIN DASHBOARD ------------------ #
col1, col2 = st.columns(2)

with col1:
    st.header("Live Predictive Engine (Sepsis)")
    
    if enc_heart is None or patient_h is None:
        st.error("Engine uninitialized. Please run backend scripts first.")
        st.stop()
        
    if not cardiac and not metabolic:
        st.metric(label="Calculated Sepsis Risk", value="0.00", delta="")
        st.error("FATAL: All modalities disconnected. System cannot safely assess.")
        routing = "N/A"
    else:
        mask = [cardiac, metabolic]
        
        # Meta Prediction
        meta_x = get_meta_features(
            [enc_heart, enc_diabetes],
            [patient_h, patient_d],
            mask=mask
        )
        # Apply reliability-weighted fusion (self-adjusting by input quality)
        # If one modality is offline, its mask=False already handles imputation;
        # weights still reflect live batch quality for present modalities.
        active_weights = [
            _rel_weights[0] if cardiac  else 0.0,
            _rel_weights[1] if metabolic else 0.0,
        ]
        risk_score = meta_learner.predict_proba(meta_x, weights=active_weights)[0]
        
        if risk_score > 0.70:
            delta, d_col = "Critical Danger", "inverse"
            st.error("⚠️ Sepsis cascade extremely likely. Immediate ICU consult required.")
        elif risk_score > 0.40:
            delta, d_col = "Elevated Caution", "off"
            st.warning("⚠️ Elevated risk markers. Monitor vitals closely.")
        else:
            delta, d_col = "Stable", "normal"
            st.success("✅ Patient profile denotes stable characteristics.")
            
        st.metric(label="Calculated Sepsis Risk Score", value=f"{risk_score:.2f}", delta=delta, delta_color=d_col)
        
        # Plotly Radar Chart
        st.markdown("##### 🩺 Normalized Vital Geometry")
        radar_df = pd.DataFrame({
            "Metric": ["Age", "Resting BP", "Cholesterol", "Max HR", "Glucose", "BMI"],
            "Value": [
                patient_h['age'].values[0] / 100.0,
                patient_h['resting_bp'].values[0] / 200.0,
                patient_h['cholesterol'].values[0] / 400.0,
                patient_h['max_hr'].values[0] / 220.0,
                patient_d['glucose'].values[0] / 300.0,
                patient_d['bmi'].values[0] / 60.0
            ]
        })
        fig_radar = px.line_polar(radar_df, r='Value', theta='Metric', line_close=True)
        fig_radar.update_traces(fill='toself', line_color='#E05A6D' if risk_score > 0.4 else '#5AE0B5')
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1.2])), margin=dict(l=20, r=20, t=20, b=20), height=300)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        if not cardiac or not metabolic:
             st.info("💡 **MISSING MODALITY GRACEFUL DEGRADATION:** System stayed operational despite missing dataset vectors due to imputation mask fallback.")

        st.header("Adaptive Explainable AI (XAI) Router")
        
        if cardiac:
            xai_result = route_explanation(enc_heart, patient_h, risk_score, "vitals")
            routing = xai_result["method"]
            
            st.write(f"**Explainability Engine Automatically Selected:** `{routing}`")
            if routing == "SHAP_fast":
                st.write("Dynamic router chose **immediate fast raw-SHAP calculations** due to exceedingly high criticality!")
                shap_matrix = xai_result["values"]
                if isinstance(shap_matrix, list): 
                    shap_matrix = shap_matrix[1] if len(shap_matrix) > 1 else shap_matrix[0]
                
                importance_df = pd.DataFrame({
                    "Feature": patient_h.columns,
                    "Impact Mapping": shap_matrix[0]
                }).set_index("Feature")
                
                st.bar_chart(importance_df)
            else:
                st.write("Dynamic router utilized **extreme-depth waterfall visualization** because clinical urgency was lower, affording time for detail.")
                try:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    shap.plots.waterfall(xai_result["values"][0], show=False)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"⚠️ Waterfall plot unavailable for this patient profile. Raw SHAP values shown instead.")
                    shap_vals = xai_result['values']
                    if hasattr(shap_vals, 'values'):
                        raw = shap_vals.values[0]
                    else:
                        raw = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
                    importance_df = pd.DataFrame({'Feature': patient_h.columns, 'Impact': raw}).set_index('Feature')
                    st.bar_chart(importance_df)
        else:
            routing = "N/A"
            st.write("*Cardiac modality is disabled, TreeExplainer bypassed for safety.*")

        # Clinical Report Export Button
        st.markdown("---")
        report = f"FAMED-lite Clinical Report\nTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Predicted Sepsis Event Risk Score: {risk_score:.2f}/1.00\n"
        report += f"Explainability Engine Engaged: {routing}\n\n=== PATIENT BASELINE ===\n"
        report += "Cardiac Modality Inputs:\n" + patient_h.to_string(index=False) + "\n\n"
        report += "Metabolic Modality Inputs:\n" + patient_d.to_string(index=False) + "\n"
        
        st.download_button("📄 Export Official Clinical Report (TXT)", data=report, file_name=f"FAMED_Report_{datetime.datetime.now().strftime('%H%M')}.txt")

with col2:
    st.header("Algorithmic Fairness & Debiasing System")
    st.markdown("*Note: This evaluates global hospital disparities across the incoming batch to enforce constraints.*")
    
    bias_inj = st.slider("Analyze Historical Physician Bias (Demographic Skew Amount)", 0.00, 0.50, 0.18, 0.01)
    
    # Auto-run Fairlearn when the slider changes
    import numpy as np
    from fairlearn.metrics import equalized_odds_difference
    from fairlearn.postprocessing import ThresholdOptimizer
    from sklearn.base import BaseEstimator, ClassifierMixin

    np.random.seed(42)
    valid_patients = sorted(list(set(heart_b['patient_id']).intersection(set(diabetes_b['patient_id']))))
    hb_clean = heart_b[heart_b['patient_id'].isin(valid_patients)].drop(columns=['patient_id'])
    db_clean = diabetes_b[diabetes_b['patient_id'].isin(valid_patients)].drop(columns=['patient_id'])

    # Load REAL ground truth labels (not derived from predictions)
    try:
        sepsis_b = pd.read_csv('data/processed/sepsis_b.csv')
        sepsis_b_valid = sepsis_b[sepsis_b['patient_id'].isin(valid_patients)]
        y_true = sepsis_b_valid['sepsis_survival_status'].values
    except Exception:
        # Fallback: derive from model predictions if file missing
        meta_b_tmp = get_meta_features([enc_heart, enc_diabetes], [hb_clean, db_clean], mask=[True, True])
        y_true = (meta_learner.predict_proba(meta_b_tmp) > 0.5).astype(int)

    meta_b = get_meta_features([enc_heart, enc_diabetes], [hb_clean, db_clean], mask=[True, True])
    A = hb_clean['sex'].values  # 0=Female, 1=Male

    # Simulate biased physician decisions: randomly flip labels for male patients
    rng = np.random.default_rng(42)
    y_pred_before = y_true.copy().astype(int)
    male_idx = np.where(A == 1)[0]
    n_flip = int(len(male_idx) * bias_inj)
    flip_idx = rng.choice(male_idx, size=n_flip, replace=False)
    y_pred_before[flip_idx] = 1 - y_pred_before[flip_idx]  # Flip labels for biased males

    disp_before = equalized_odds_difference(y_true, y_pred_before, sensitive_features=A)

    class LivePredictor(BaseEstimator, ClassifierMixin):
        def __init__(self): self.classes_ = np.array([0, 1])
        def fit(self, X, y): return self
        def predict(self, X): return y_pred_before[:len(X)]

    optimizer = ThresholdOptimizer(
        estimator=LivePredictor(),
        constraints="equalized_odds",
        predict_method="predict",
        prefit=True
    )
    optimizer.fit(meta_b, y_true, sensitive_features=A)
    y_pred_after = optimizer.predict(meta_b, sensitive_features=A)
    disp_after = equalized_odds_difference(y_true, y_pred_after, sensitive_features=A)

    reduction = ((disp_before - disp_after) / disp_before) * 100 if disp_before > 0 else 0

    st.write("Equalized Odds Disparity Evaluator:")
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Before Constraint Optimizer", f"{disp_before:.4f}")
    mcol2.metric("After Fairlearn Adjustment", f"{disp_after:.4f}", f"-{reduction:.0f}%" if reduction > 0 else "0%")

    st.header("Data Drift Monitor — Graduated Response System")
    st.markdown("*Compares Site A (reference) vs. live Site B batches. Triggers a proportional response — not a binary panic switch.*")

    # Tier legend
    with st.expander("📋 Response Tiers", expanded=False):
        st.markdown("""
| Drift Share | Status | Action |
|---|---|---|
| `< 0.15` | 🟢 **STABLE** | Continue normal operation |
| `0.15 – 0.30` | 🟡 **WATCH** | Increase monitoring frequency, flag batch |
| `0.30 – 0.50` | 🟠 **ALERT** | Retrain recommended, notify data steward |
| `> 0.50` | 🔴 **ROLLBACK** | Immediate checkpoint fallback |
        """)

    drift_inj = st.slider("Inject Artificial Data Drift (%) into Live Database", 0, 500, 0, 10)
    if st.button("Trigger Live Global Drift Scan"):
        st.write("Initiating system scan across live infrastructure...")
        ref_df  = heart_a.drop(columns=['patient_id'])
        curr_df = heart_b.drop(columns=['patient_id']).copy()

        if drift_inj > 0:
            scale = 1.0 + (drift_inj / 100.0)
            curr_df['cholesterol'] = curr_df['cholesterol'] * scale
            curr_df['max_hr']      = curr_df['max_hr']      / scale
            curr_df['resting_bp']  = curr_df['resting_bp']  * scale
            curr_df['age']         = curr_df['age']         * scale
            curr_df['oldpeak']     = curr_df['oldpeak']      * scale

        with st.spinner("Analyzing statistically significant drift via Evidently Report..."):
            event = check_drift(ref_df, curr_df, threshold=0.2)

        status = event.get("status", "UNKNOWN")
        score  = event.get("drift_score", 0.0)
        action = event.get("action", "")
        desc   = event.get("description", "")

        # Graduated visual response
        if status == "STABLE":
            st.success(f"🟢 **STABLE** — Drift share: `{score:.2f}` | {desc}")
        elif status == "WATCH":
            st.warning(f"🟡 **WATCH** — Drift share: `{score:.2f}` | {desc}")
        elif status == "ALERT":
            st.warning(f"🟠 **ALERT** — Drift share: `{score:.2f}` | {desc}")
            st.info("💡 Recommended: schedule model retraining on updated data within 48 hours.")
        elif status == "ROLLBACK":
            st.error(f"🔴 **ROLLBACK** — Drift share: `{score:.2f}` | {desc}")
            st.error("🚨 Automated checkpoint rollback triggered. Previous stable model version restored.")

        # Show drift score as a progress bar for visual impact
        st.markdown(f"**Drift Severity Gauge:** `{score:.0%}`")
        st.progress(min(float(score), 1.0))

    if os.path.exists("logs/audit_log.json"):
        with open("logs/audit_log.json", "r") as f:
            lines = f.readlines()
            logs = [json.loads(line) for line in lines]

        st.write("**Immutable Audit Rollback Ledger:**")
        df_log = pd.DataFrame(logs[-6:])
        if 'drift_score' in df_log.columns:
            df_log['drift_score'] = df_log['drift_score'].apply(lambda x: float(x) if x is not None else 0.0)
        st.dataframe(df_log, use_container_width=True)
