import streamlit as st
import os
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from gradcam import make_gradcam_heatmap, save_and_display_gradcam
from fpdf import FPDF
import io
from datetime import datetime

# ---------------------------------------------------------
# STREAMLIT PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(page_title="DR Classifier", layout="wide")
st.title("👁️ Diabetic Retinopathy Detector")


# ---------------------------------------------------------
# GOOGLE DRIVE MODEL DOWNLOAD LINKS
# ---------------------------------------------------------
MODEL_URLS = {
    "best_model.h5": "https://drive.google.com/uc?id=1yKtQdHNaVFIq6g-j6Mn0RlqYTS5RjXre",
    "final_model.h5": "https://drive.google.com/uc?id=1T2cgXyayzJ4eskeSX2oE4_QLyeNU8BjM",
    "final1.h5": "https://drive.google.com/uc?id=1zV42a1RjybxQ3dnAmT6alizpwOED6PmB"
}



# ---------------------------------------------------------
# FUNCTION: DOWNLOAD MODEL IF MISSING
# ---------------------------------------------------------
def ensure_model(model_name):
    model_path = f"models/{model_name}"

    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)

    if not os.path.exists(model_path):
        st.warning(f"⬇ Downloading model: {model_name} ... please wait")
        gdown.download(url=MODEL_URLS[model_name], output=model_path, quiet=False, fuzzy=True)
        st.success(f"Model {model_name} downloaded successfully!")

    return model_path


# ---------------------------------------------------------
# SELECT MODEL
# ---------------------------------------------------------
st.subheader("Select Deep Learning Model")

selected_model = st.selectbox("Choose a model:", list(MODEL_URLS.keys()))

MODEL_PATH = ensure_model(selected_model)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_dr_model(path):
    return load_model(path, compile=False)


model = load_dr_model(MODEL_PATH)
CLASS_NAMES = ["Healthy", "Mild DR", "Moderate DR", "Proliferate DR", "Severe DR"]


# ---------------------------------------------------------
# SIDEBAR: PATIENT DETAILS
# ---------------------------------------------------------
st.sidebar.header("Patient Details")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
notes = st.sidebar.text_area("Notes (optional)")


# ---------------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload retinal fundus images",
    accept_multiple_files=True,
    type=['png','jpg','jpeg']
)

results = []
image_pairs = []


# ---------------------------------------------------------
# PROCESS IMAGES
# ---------------------------------------------------------
if uploaded_files:
    for file in uploaded_files:
        st.markdown(f"### 🔍 Processing: **{file.name}**")

        img = Image.open(file).convert('RGB')
        tmp_path = "temp_input.jpg"
        img.save(tmp_path)

        input_shape = model.input_shape[1:3]

        arr = image.load_img(tmp_path, target_size=input_shape)
        arr = image.img_to_array(arr) / 255.0
        x = np.expand_dims(arr, axis=0)

        preds = model.predict(x)
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]) * 100)
        pred_class = CLASS_NAMES[idx]

        st.success(f"Prediction: **{pred_class}** ({conf:.2f}% confidence)")

        results.append({
            "filename": file.name,
            "class": pred_class,
            "confidence": conf
        })

        # ---------------------------------------------------------
        # GRAD-CAM
        # ---------------------------------------------------------
        last_conv = None
        for layer in reversed(model.layers):
            try:
                if len(layer.output_shape) == 4:
                    last_conv = layer.name
                    break
            except:
                continue

        heatmap = make_gradcam_heatmap(x, model, last_conv, pred_index=idx)
        cam_img = save_and_display_gradcam(
            tmp_path, heatmap, cam_path="gradcam_temp.jpg",
            alpha=0.4, size=input_shape
        )
        cam_img = np.asarray(cam_img)

        if cam_img.dtype != np.uint8:
            cam_img = cam_img - cam_img.min()
            cam_img = cam_img / (cam_img.max() + 1e-9)
            cam_img = (cam_img * 255).astype(np.uint8)

        col1, col2 = st.columns(2)
        col1.image(img, caption=f"Input Image: {file.name}", width=350)
        col2.image(cam_img, caption=f"Grad-CAM: {file.name}", width=350)

        image_pairs.append((img, Image.fromarray(cam_img), file.name))

        st.markdown("---")


    # ---------------------------------------------------------
    # SUMMARY TABLE
    # ---------------------------------------------------------
    st.subheader("📊 Summary of Predictions")
    st.table(results)


    # ---------------------------------------------------------
    # GENERATE PDF REPORT
    # ---------------------------------------------------------
    if st.button("📥 Generate PDF Report"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=14)

        pdf.cell(0, 10, "Diabetic Retinopathy Prediction Report", ln=True, align='C')
        pdf.ln(5)

        today = datetime.now().strftime("%d-%m-%Y")
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, f"Date: {today}", ln=True)
        pdf.cell(0, 8, f"Patient: {name} | Age: {age} | Gender: {gender}", ln=True)
        pdf.multi_cell(0, 8, f"Notes: {notes}")
        pdf.ln(5)

        for r in results:
            pdf.cell(0, 8, f"{r['filename']} - {r['class']} ({r['confidence']:.2f}%)", ln=True)

        pdf.ln(10)
        pdf.cell(0, 10, "Images & Grad-CAM", ln=True)

        for input_img, grad_img, fname in image_pairs:

            if pdf.get_y() > 210:
                pdf.add_page()

            pdf.cell(0, 8, f"Image: {fname}", ln=True)

            left_img = "left_tmp.jpg"
            right_img = "right_tmp.jpg"

            input_img.save(left_img, "JPEG")
            grad_img.save(right_img, "JPEG")

            pdf.image(left_img, x=15, y=pdf.get_y()+5, w=80)
            pdf.image(right_img, x=115, y=pdf.get_y()+5, w=80)

            pdf.set_y(pdf.get_y() + 90)
            pdf.ln(10)

        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        buffer = io.BytesIO(pdf_bytes)
        buffer.seek(0)

        st.download_button(
            "Download PDF Report",
            data=buffer,
            file_name=f"{name if name else 'patient'}_DR_Report.pdf",
            mime="application/pdf"
        )


else:
    st.info("Upload retinal images to begin predictions.")
