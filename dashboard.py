import streamlit as st
import os
import numpy as np
import gdown
import pandas as pd
import uuid
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from gradcam import make_gradcam_heatmap, save_and_display_gradcam
from fpdf import FPDF
import io
from datetime import datetime

# ---------------------------------------------------------
# INITIALIZE SESSION STATE
# ---------------------------------------------------------
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = f"DR-{uuid.uuid4().hex[:8].upper()}"
if 'locked' not in st.session_state:
    st.session_state.locked = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'form_iteration' not in st.session_state:
    st.session_state.form_iteration = 0

# ---------------------------------------------------------
# CONSTANTS & MAPPINGS
# ---------------------------------------------------------
CLASS_NAMES = ["Healthy", "Mild DR", "Moderate DR", "Severe DR", "Proliferate DR"]

CLINICAL_REC = {
    "Healthy": "Routine annual screening recommended.",
    "Mild DR": "Monitor closely. Follow-up in 6-12 months. Control blood sugar levels.",
    "Moderate DR": "Referral to ophthalmologist. Possible treatment may be required soon.",
    "Severe DR": "Urgent referral required. High risk of vision loss. Immediate intervention.",
    "Proliferate DR": "Medical Emergency. Immediate laser surgery or injections likely needed."
}

# ---------------------------------------------------------
# PAGE CONFIG & CSS
# ---------------------------------------------------------
st.set_page_config(page_title="DR Detector Pro", layout="wide")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp { background-color: #0e1117; color: white; }
    .hero { background: linear-gradient(135deg, #6366f1, #9333ea); padding: 40px; border-radius: 18px; text-align: center; margin-bottom: 25px; }
    .card { background: #161b22; padding: 22px; border-radius: 16px; margin-bottom: 20px; border: 1px solid #30363d; }
    .stats { display: flex; gap: 20px; margin-top: 15px; }
    .stat { flex: 1; background: #111827; padding: 15px; border-radius: 14px; text-align: center; border: 1px solid #30363d; }
    .stat h2 { color: #60a5fa; margin: 0; }
    .badge { padding: 6px 16px; border-radius: 999px; font-weight: bold; display: inline-block; margin-bottom: 10px; }
    .red { background: #dc2626; } .yellow { background: #facc15; color: black; } .green { background: #22c55e; }
    .rec-box { background: #111827; border-left: 5px solid #6366f1; padding: 15px; border-radius: 8px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <h1>👁️ Diabetic Retinopathy Detector</h1>
    <p>AI-powered system for early screening and severity classification</p>
    <p><b>Project Lead:</b> Manendra Singh</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# ABOUT SECTION
# ---------------------------------------------------------
st.markdown("""
<div class="card">
    <h3>About Diabetic Retinopathy</h3>
    <p>Diabetic retinopathy is a complication of diabetes that affects the eyes. It is caused by damage to the retinal blood vessels.</p>
    <div class="stats">
      <div class="stat"><h2>830M</h2><p>Diabetes Cases</p></div>
      <div class="stat"><h2>126M</h2><p>DR Cases Global</p></div>
      <div class="stat"><h2>AI</h2><p>Fast Screening</p></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------
MODEL_URLS = {
    "best_model.h5": "https://drive.google.com/file/d/1yKtQdHNaVFIq6g-j6Mn0RlqYTS5RjXre/view?usp=sharing",
    "final_model.h5": "https://drive.google.com/file/d/1T2cgXyayzJ4eskeSX2oE4_QLyeNU8BjM/view?usp=sharing",
    "final1.h5": "https://drive.google.com/file/d/1zV42a1RjybxQ3dnAmT6alizpwOED6PmB/view?usp=sharing"
}

def ensure_model(model_name):
    os.makedirs("models", exist_ok=True)
    path = f"models/{model_name}"
    if not os.path.exists(path):
        with st.spinner(f"Downloading model: {model_name}..."):
            gdown.download(MODEL_URLS[model_name], path, quiet=False)
    return path

st.markdown('<div class="card"><h3>🧠 Select Deep Learning Engine</h3>', unsafe_allow_html=True)
selected_model = st.selectbox("Choose architecture:", list(MODEL_URLS.keys()), key=f"model_sel_{st.session_state.form_iteration}")
MODEL_PATH = ensure_model(selected_model)
st.markdown('</div>', unsafe_allow_html=True)

@st.cache_resource
def load_dr_model(path):
    return load_model(path, compile=False)

model = load_dr_model(MODEL_PATH)

# ---------------------------------------------------------
# PATIENT DETAILS (With Reset Keys)
# ---------------------------------------------------------
st.markdown('<div class="card"><h3>👤 Patient Intake Form</h3>', unsafe_allow_html=True)
c_id, c1, c2, c3 = st.columns([1.5, 2, 1, 1])

# The key=f"..._{st.session_state.form_iteration}" forces a reset when iteration increases
c_id.text_input("System Patient ID", value=st.session_state.patient_id, disabled=True, key=f"id_box_{st.session_state.form_iteration}")
p_name = c1.text_input("Full Name", disabled=st.session_state.locked, key=f"name_{st.session_state.form_iteration}")
p_age = c2.number_input("Age", 0, 120, 30, disabled=st.session_state.locked, key=f"age_{st.session_state.form_iteration}")
p_gender = c3.selectbox("Gender", ["Male", "Female", "Other"], disabled=st.session_state.locked, key=f"gen_{st.session_state.form_iteration}")

p_notes = st.text_area("Clinical History/Notes", disabled=st.session_state.locked, key=f"notes_{st.session_state.form_iteration}")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------
st.markdown('<div class="card"><h3>📁 Upload Fundus Images</h3>', unsafe_allow_html=True)
uploaded_files = st.file_uploader("Select retinal fundus images", accept_multiple_files=True, type=["png", "jpg", "jpeg"], key=f"uploader_{st.session_state.form_iteration}")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_files:
    st.session_state.locked = True
    temp_results = []

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        img.save("temp.jpg")
        
        input_shape = model.input_shape[1:3]
        arr = image.load_img("temp.jpg", target_size=input_shape)
        prep_arr = image.img_to_array(arr) / 255.0
        x = np.expand_dims(prep_arr, axis=0)

        preds = model.predict(x)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx] * 100)
        pred_class = CLASS_NAMES[idx]

        with st.expander(f"Detailed Analysis: {file.name}", expanded=True):
            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.image(img, caption="Input Image", use_container_width=True)
                chart_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": preds})
                st.bar_chart(chart_df.set_index("Class"))

            with col_right:
                color = "red" if idx >= 3 else "yellow" if idx >= 1 else "green"
                st.markdown(f"### Result: <span class='badge {color}'>{pred_class}</span>", unsafe_allow_html=True)
                st.metric("Confidence Score", f"{conf:.2f}%")
                st.markdown(f'<div class="rec-box"><b>Recommendation:</b><br>{CLINICAL_REC[pred_class]}</div>', unsafe_allow_html=True)

                try:
                    last_conv = next(l.name for l in reversed(model.layers) if len(l.output_shape) == 4)
                    heatmap = make_gradcam_heatmap(x, model, last_conv, idx)
                    cam_img = save_and_display_gradcam("temp.jpg", heatmap, "cam.jpg", size=input_shape)
                    cam_arr = np.array(cam_img)
                    cam_fixed = ((cam_arr - cam_arr.min()) * (255 / (cam_arr.max() - cam_arr.min() + 1e-8))).astype(np.uint8)
                    st.image(cam_fixed, caption="Pathology Localization (Grad-CAM)", use_container_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM error: {e}")

            temp_results.append({
                "Filename": file.name, "Prediction": pred_class, 
                "Confidence": f"{conf:.2f}%", "Recommendation": CLINICAL_REC[pred_class]
            })

    st.session_state.analysis_results = temp_results

    # ---------------------------------------------------------
    # SUMMARY & EXPORTS
    # ---------------------------------------------------------
    st.markdown('<div class="card"><h3>📊 Batch Analysis Summary</h3>', unsafe_allow_html=True)
    summary_df = pd.DataFrame(st.session_state.analysis_results)
    st.table(summary_df)

    ec1, ec2, ec3 = st.columns(3)
    
    csv = summary_df.to_csv(index=False).encode('utf-8')
    ec1.download_button("📥 Download CSV", data=csv, file_name=f"Batch_{st.session_state.patient_id}.csv", mime="text/csv")

    if ec2.button("📄 Generate Medical PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Diabetic Retinopathy Screening Report", ln=True, align='C')
        pdf.set_font("Arial", size=10)
        pdf.ln(5)
        pdf.cell(0, 7, f"Patient ID: {st.session_state.patient_id}", ln=True)
        pdf.cell(0, 7, f"Name: {p_name} | Age: {p_age} | Gender: {p_gender}", ln=True)
        pdf.cell(0, 7, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(5); pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(5)
        for r in st.session_state.analysis_results:
            pdf.set_font("Arial", 'B', 11); pdf.cell(0, 7, f"Image: {r['Filename']}", ln=True)
            pdf.set_font("Arial", size=10); pdf.cell(0, 6, f"Diagnosis: {r['Prediction']} ({r['Confidence']})", ln=True)
            pdf.multi_cell(0, 6, f"Recommendation: {r['Recommendation']}"); pdf.ln(4)
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        st.download_button("Download PDF", data=pdf_bytes, file_name=f"Report_{st.session_state.patient_id}.pdf")

    # START NEW PATIENT TRIGGER
    if ec3.button("🔄 Start New Patient"):
        st.session_state.locked = False
        st.session_state.patient_id = f"DR-{uuid.uuid4().hex[:8].upper()}"
        st.session_state.analysis_results = []
        st.session_state.form_iteration += 1  # This forces all widgets to reset
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Awaiting fundus image upload for analysis.")

st.markdown("""
<br><br>
<div style="background: #1f2937; border-left: 6px solid #facc15; text-align: center; color: #d1d5db; padding: 20px;">
    <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 18px;">
        ⚠️ Important Medical Disclaimer
    </h3>
    <p style="font-size: 14px; line-height: 1.6; max-width: 800px; margin: 0 auto;">
        This AI system is designed as a screening and education tool. It is not intended to diagnose, treat, cure, or prevent any disease.
        Results should not replace professional medical advice, diagnosis, or treatment.<br>
        <br>
        <span style="color: #f3f4f6; font-weight: 500;">Always consult with a qualified healthcare professional for medical advice and treatment decisions.</span>
    </p>
    <div style="margin-top: 25px; font-size: 13px; color: #9ca3af; display: flex; justify-content: center; gap: 20px; align-items: center; flex-wrap: wrap;">
        <span>🤖 Powered by Advanced AI</span>
        <span style="color: #4b5563;">|</span>
        <span>🔬 Medical Grade Analysis</span>
        <span style="color: #4b5563;">|</span>
        <span>🌐 Accessible Health Technology</span>
    </div>
</div>
""", unsafe_allow_html=True)
